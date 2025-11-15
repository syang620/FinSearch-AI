#!/usr/bin/env python3
"""
Ingest EDGAR filings from data/edgar/ into ChromaDB vector database
with BGE-base-en-v1.5 embeddings
"""

import os
import sys
import logging
from pathlib import Path
from glob import glob

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.services.rag.retriever import rag_retriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_edgar_files(data_dir: str = "data/edgar"):
    """
    Ingest all EDGAR filing .txt files into vector database

    Args:
        data_dir: Root directory containing company folders with filings
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Find all .txt files
    txt_files = list(data_path.glob("*/*.txt"))

    logger.info(f"Found {len(txt_files)} EDGAR filing files to ingest")
    logger.info(f"Using embedding model: BGE-base-en-v1.5 (768-dim)")

    # Track results
    successful = 0
    failed = 0
    total_chunks = 0

    # Process each file
    for i, file_path in enumerate(txt_files, 1):
        try:
            # Extract metadata from path
            # e.g., data/edgar/AAPL/10-K_2025.txt
            parts = file_path.parts
            company = parts[-2]  # AAPL
            filename = parts[-1]  # 10-K_2025.txt

            # Parse filing type and year
            filing_parts = filename.replace('.txt', '').split('_')
            filing_type = filing_parts[0]  # 10-K or 10-Q
            year = filing_parts[1] if len(filing_parts) > 1 else "unknown"

            metadata = {
                "company": company,
                "document_type": filing_type,
                "year": year,
                "filename": filename,
                "source": "edgar"
            }

            logger.info(f"[{i}/{len(txt_files)}] Ingesting {company} {filing_type} {year}...")

            # Ingest the document
            result = rag_retriever.ingest_document(
                file_path=str(file_path),
                metadata=metadata
            )

            if result["success"]:
                successful += 1
                total_chunks += result["chunks_created"]
                logger.info(f"  ✓ Created {result['chunks_created']} chunks")
            else:
                failed += 1
                logger.error(f"  ✗ Failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            failed += 1
            logger.error(f"  ✗ Error processing {file_path}: {e}")

    # Print summary
    print("\n" + "="*70)
    print("VECTOR DATABASE INGESTION COMPLETE")
    print("="*70)
    print(f"Files processed: {successful}/{len(txt_files)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Failed: {failed}")
    print(f"\nEmbedding model: BGE-base-en-v1.5 (768-dim)")
    print(f"Vector database: ChromaDB")
    print(f"Collection: financial_documents")
    print("="*70)

    # Get collection stats
    try:
        stats = rag_retriever.get_stats()
        print(f"\nVector Store Stats:")
        print(f"  Total documents: {stats.get('count', 0)}")
    except Exception as e:
        logger.error(f"Could not get stats: {e}")


if __name__ == "__main__":
    logger.info("Starting EDGAR filings ingestion to vector database")
    ingest_edgar_files()
    logger.info("Ingestion complete")
