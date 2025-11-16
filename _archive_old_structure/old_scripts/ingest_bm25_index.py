#!/usr/bin/env python3
"""
Ingest chunked documents into BM25 index (SQLite FTS5)
This complements the dense vector embeddings for hybrid search
"""

import json
import sys
import os
from pathlib import Path
import logging
from typing import List, Dict, Any
from tqdm import tqdm

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.services.rag.bm25_index import bm25_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bm25_indexing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def read_chunked_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read chunks from a JSONL file."""
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []


def ingest_all_chunks(data_dir: str = "data_chunked"):
    """Ingest all chunked JSONL files into BM25 index."""
    logger.info("Starting BM25 indexing of chunked documents")

    # Find all *_chunks.jsonl files
    chunk_files = list(Path(data_dir).rglob("*_chunks.jsonl"))
    logger.info(f"Found {len(chunk_files)} chunked files to index")

    if not chunk_files:
        logger.error("No chunked files found!")
        return

    # Clear existing index
    logger.info("Clearing existing BM25 index...")
    bm25_index.clear()

    total_indexed = 0
    failed_files = []

    # Process each file
    with tqdm(total=len(chunk_files), desc="Indexing files") as pbar:
        for file_path in chunk_files:
            pbar.set_description(f"Indexing {file_path.name}")

            # Read chunks
            chunks = read_chunked_jsonl(str(file_path))
            if not chunks:
                logger.warning(f"No chunks found in {file_path}")
                failed_files.append(str(file_path))
                pbar.update(1)
                continue

            # Prepare documents for BM25
            documents = []
            for chunk in chunks:
                doc = {
                    'chunk_id': chunk.get('chunk_id'),
                    'text': chunk.get('text', ''),
                    'ticker': chunk.get('ticker'),
                    'doc_type': chunk.get('doc_type'),
                    'fiscal_year': chunk.get('fiscal_year'),
                    'quarter': chunk.get('quarter'),
                    'phase': chunk.get('phase'),
                    'speaker': chunk.get('speaker'),
                    'speaker_role': chunk.get('speaker_role'),
                    'section_title': chunk.get('section_title'),
                    'chunk_tokens': chunk.get('chunk_tokens', 0)
                }
                documents.append(doc)

            # Index documents
            try:
                indexed_count = bm25_index.add_documents(documents)
                total_indexed += indexed_count
                logger.info(f"Indexed {indexed_count} chunks from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to index {file_path}: {e}")
                failed_files.append(str(file_path))

            pbar.update(1)

    # Print summary
    print("\n" + "="*70)
    print("BM25 INDEXING COMPLETE")
    print("="*70)
    print(f"Files processed: {len(chunk_files)}")
    print(f"Total chunks indexed: {total_indexed}")
    print(f"Failed files: {len(failed_files)}")
    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    print("="*70)

    # Verify index
    doc_count = bm25_index.count()
    logger.info(f"BM25 index now contains {doc_count} documents")

    # Test search
    logger.info("\nTesting BM25 search...")
    test_queries = [
        "Item 1A Risk Factors",
        "AAPL revenue growth",
        "Tim Cook CEO",
        "ASC 606",
        "10-K 2024"
    ]

    for query in test_queries:
        results = bm25_index.search(query, n_results=3)
        doc_count = len(results['ids'][0]) if results['ids'] else 0
        logger.info(f"Query: '{query}' -> {doc_count} results")
        if doc_count > 0 and results['metadatas'][0]:
            first_result = results['metadatas'][0][0]
            logger.info(f"  Top result: {first_result.get('ticker')} {first_result.get('doc_type')}")


if __name__ == "__main__":
    ingest_all_chunks()