"""
BM25 Index Service for Keyword/Lexical Search using SQLite FTS5
Complements the dense vector search for hybrid retrieval
"""

import sqlite3
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class BM25IndexService:
    """SQLite FTS5-based BM25 index for keyword search"""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize BM25 index with SQLite FTS5"""
        if db_path is None:
            # Use the same directory as ChromaDB
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "../../../..", "data")
            db_path = os.path.join(data_dir, "bm25_index.db")

        self.db_path = db_path
        self.conn = None
        self._initialize_db()

    def _initialize_db(self):
        """Create FTS5 table if not exists"""
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()

            # Create FTS5 virtual table for full-text search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    chunk_id UNINDEXED,
                    content,
                    ticker,
                    doc_type,
                    fiscal_year UNINDEXED,
                    quarter UNINDEXED,
                    phase UNINDEXED,
                    speaker UNINDEXED,
                    tokenize='porter'
                )
            """)

            # Create metadata table for additional filtering
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    chunk_id TEXT PRIMARY KEY,
                    ticker TEXT,
                    doc_type TEXT,
                    fiscal_year INTEGER,
                    quarter TEXT,
                    phase TEXT,
                    speaker TEXT,
                    speaker_role TEXT,
                    section_title TEXT,
                    chunk_tokens INTEGER
                )
            """)

            # Create indexes for efficient filtering
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker
                ON document_metadata(ticker)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_type
                ON document_metadata(doc_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_fiscal_year
                ON document_metadata(fiscal_year)
            """)

            self.conn.commit()
            logger.info(f"BM25 index initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents to BM25 index

        Args:
            documents: List of document dicts with 'chunk_id', 'text', and metadata

        Returns:
            Number of documents indexed
        """
        try:
            cursor = self.conn.cursor()
            indexed_count = 0

            for doc in documents:
                # Extract required fields
                chunk_id = doc.get('chunk_id')
                text = doc.get('text', '')
                ticker = doc.get('ticker', '')
                doc_type = doc.get('doc_type', '')

                if not chunk_id or not text:
                    continue

                # Insert into FTS5 table
                cursor.execute("""
                    INSERT OR REPLACE INTO documents_fts
                    (chunk_id, content, ticker, doc_type, fiscal_year, quarter, phase, speaker)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id,
                    text,
                    ticker,
                    doc_type,
                    str(doc.get('fiscal_year', '')),
                    doc.get('quarter', ''),
                    doc.get('phase', ''),
                    doc.get('speaker', '')
                ))

                # Insert into metadata table
                cursor.execute("""
                    INSERT OR REPLACE INTO document_metadata
                    (chunk_id, ticker, doc_type, fiscal_year, quarter, phase,
                     speaker, speaker_role, section_title, chunk_tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id,
                    ticker,
                    doc_type,
                    doc.get('fiscal_year'),
                    doc.get('quarter'),
                    doc.get('phase'),
                    doc.get('speaker'),
                    doc.get('speaker_role'),
                    doc.get('section_title'),
                    doc.get('chunk_tokens', 0)
                ))

                indexed_count += 1

            self.conn.commit()
            logger.info(f"Indexed {indexed_count} documents in BM25")
            return indexed_count

        except Exception as e:
            logger.error(f"Error adding documents to BM25 index: {e}")
            self.conn.rollback()
            raise

    def search(self,
               query: str,
               n_results: int = 10,
               where: Optional[Dict[str, Any]] = None) -> Dict[str, List]:
        """
        Search using BM25 ranking

        Args:
            query: Search query
            n_results: Number of results to return
            where: Optional filter conditions

        Returns:
            Dict with 'ids', 'scores', 'documents', 'metadatas'
        """
        try:
            cursor = self.conn.cursor()

            # Build the query
            sql_query = """
                SELECT
                    f.chunk_id,
                    f.content,
                    -f.rank as score,
                    m.ticker,
                    m.doc_type,
                    m.fiscal_year,
                    m.quarter,
                    m.phase,
                    m.speaker,
                    m.speaker_role,
                    m.section_title
                FROM documents_fts f
                JOIN document_metadata m ON f.chunk_id = m.chunk_id
                WHERE documents_fts MATCH ?
            """

            # Escape special characters in FTS5 query
            # Replace apostrophes and commas with spaces to avoid syntax errors
            query_escaped = query.replace("'", " ").replace(",", " ")
            # Wrap each word in quotes for FTS5 to handle properly
            # This prevents syntax errors with special characters
            words = query_escaped.split()
            if words:
                # Use FTS5 phrase query for better results
                query_formatted = ' '.join([f'"{word}"' for word in words])
            else:
                query_formatted = '""'
            params = [query_formatted]

            # Add filters if provided
            if where:
                conditions = []
                for key, value in where.items():
                    if isinstance(value, dict):  # Handle operators like $gte
                        for op, val in value.items():
                            if op == "$gte":
                                conditions.append(f"m.{key} >= ?")
                                params.append(val)
                            elif op == "$lte":
                                conditions.append(f"m.{key} <= ?")
                                params.append(val)
                            elif op == "$in":
                                placeholders = ','.join(['?' for _ in val])
                                conditions.append(f"m.{key} IN ({placeholders})")
                                params.extend(val)
                    else:
                        conditions.append(f"m.{key} = ?")
                        params.append(value)

                if conditions:
                    sql_query += " AND " + " AND ".join(conditions)

            sql_query += f" ORDER BY rank LIMIT {n_results}"

            # Execute search
            cursor.execute(sql_query, params)
            results = cursor.fetchall()

            # Format results similar to ChromaDB output
            ids = []
            documents = []
            scores = []
            metadatas = []

            for row in results:
                ids.append(row[0])
                documents.append(row[1])
                scores.append(row[2])  # BM25 score

                metadata = {
                    'ticker': row[3],
                    'doc_type': row[4],
                    'fiscal_year': row[5],
                    'quarter': row[6],
                    'phase': row[7],
                    'speaker': row[8],
                    'speaker_role': row[9],
                    'section_title': row[10]
                }
                # Remove None values
                metadata = {k: v for k, v in metadata.items() if v is not None}
                metadatas.append(metadata)

            return {
                'ids': [ids],  # Nested list to match ChromaDB format
                'documents': [documents],
                'scores': [scores],
                'metadatas': [metadatas]
            }

        except Exception as e:
            logger.error(f"Error searching BM25 index: {e}")
            return {'ids': [[]], 'documents': [[]], 'scores': [[]], 'metadatas': [[]]}

    def count(self) -> int:
        """Get total document count"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM document_metadata")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0

    def clear(self):
        """Clear all documents from index"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM documents_fts")
            cursor.execute("DELETE FROM document_metadata")
            self.conn.commit()
            logger.info("BM25 index cleared")
        except Exception as e:
            logger.error(f"Error clearing BM25 index: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


# Singleton instance
bm25_index = BM25IndexService()