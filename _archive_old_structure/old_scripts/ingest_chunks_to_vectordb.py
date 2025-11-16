#!/usr/bin/env python3
"""
Ingest chunked documents (earnings transcripts and EDGAR filings) into ChromaDB
vector database with BGE-base-en-v1.5 embeddings.

This script reads pre-chunked JSONL files from data_chunked/ directory and ingests
them into the vector database with full metadata preservation.
"""

import os
import sys
import json
import logging
from pathlib import Path
from glob import glob
from typing import List, Dict, Any

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.services.rag.vector_store import vector_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_chunked_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Read chunks from a JSONL file.

    Args:
        file_path: Path to chunked JSONL file

    Returns:
        List of chunk dictionaries
    """
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def prepare_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and prepare metadata from chunk for vector store.

    Args:
        chunk: Chunk dictionary

    Returns:
        Metadata dictionary for vector store
    """
    # Base metadata
    metadata = {
        "chunk_id": chunk.get("chunk_id"),
        "doc_id": chunk.get("doc_id"),
        "ticker": chunk.get("ticker"),
        "company": chunk.get("company"),
        "doc_type": chunk.get("doc_type"),
        "source_uri": chunk.get("source_uri"),
        "chunk_type": chunk.get("chunk_type"),
        "chunk_tokens": chunk.get("chunk_tokens", 0),
    }

    # Document type specific metadata
    if chunk.get("doc_type") == "earnings_transcript":
        # Earnings call metadata
        metadata.update({
            "fiscal_year": chunk.get("fiscal_year"),
            "quarter": chunk.get("quarter"),
            "period": chunk.get("period"),
            "phase": chunk.get("phase"),  # prepared_remarks or qa
            "speaker": chunk.get("speaker"),
            "speaker_role": chunk.get("speaker_role"),
        })
    elif chunk.get("doc_type") in ["10-K", "10-Q"]:
        # EDGAR filing metadata
        metadata.update({
            "fiscal_year": chunk.get("fiscal_year"),
            "section_id": chunk.get("section_id"),
            "section_title": chunk.get("section_title"),
        })

    # Remove None values
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return metadata


def ingest_chunked_file(file_path: str) -> Dict[str, Any]:
    """
    Ingest a chunked JSONL file into vector database.

    Args:
        file_path: Path to chunked JSONL file

    Returns:
        Result dictionary with success status and counts
    """
    try:
        # Read chunks
        chunks = read_chunked_jsonl(file_path)

        if not chunks:
            logger.warning(f"No chunks found in {file_path}")
            return {"success": False, "error": "No chunks found"}

        # Prepare texts and metadatas for batch ingestion
        texts = []
        metadatas = []

        for chunk in chunks:
            # Skip very small chunks (likely validation errors)
            if chunk.get("chunk_tokens", 0) < 50:
                continue

            text = chunk.get("text", "")
            if not text:
                continue

            metadata = prepare_metadata(chunk)

            texts.append(text)
            metadatas.append(metadata)

        if not texts:
            logger.warning(f"No valid chunks to ingest from {file_path}")
            return {"success": False, "error": "No valid chunks"}

        # Add to vector store
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        chunk_ids = vector_store.add_documents(texts, metadatas)

        logger.info(f"Successfully ingested {len(chunk_ids)} chunks")

        return {
            "success": True,
            "chunks_ingested": len(chunk_ids),
            "total_chunks": len(chunks)
        }

    except Exception as e:
        logger.error(f"Error ingesting {file_path}: {e}")
        return {"success": False, "error": str(e)}


def ingest_all_chunks(data_dir: str = "data_chunked"):
    """
    Ingest all chunked JSONL files from data_chunked directory.

    Args:
        data_dir: Root directory containing chunked files
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Find all chunked JSONL files
    chunk_files = list(data_path.rglob("*_chunks.jsonl"))

    logger.info(f"Found {len(chunk_files)} chunked files to ingest")
    logger.info(f"Using embedding model: BGE-base-en-v1.5 (768-dim)")

    # Track results
    successful = 0
    failed = 0
    total_chunks_ingested = 0

    # Process each file
    for i, file_path in enumerate(chunk_files, 1):
        try:
            # Extract info from path for logging
            parts = file_path.parts
            ticker = parts[-3]  # e.g., AAPL
            doc_type_dir = parts[-2]  # edgar or earnings
            filename = parts[-1]

            logger.info(f"[{i}/{len(chunk_files)}] Ingesting {ticker}/{doc_type_dir}/{filename}...")

            # Ingest the file
            result = ingest_chunked_file(str(file_path))

            if result["success"]:
                successful += 1
                chunks_count = result["chunks_ingested"]
                total_chunks_ingested += chunks_count
                logger.info(f"  ✓ Ingested {chunks_count} chunks")
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
    print(f"Files processed: {successful}/{len(chunk_files)}")
    print(f"Total chunks ingested: {total_chunks_ingested:,}")
    print(f"Failed: {failed}")
    print(f"\nEmbedding model: BGE-base-en-v1.5 (768-dim)")
    print(f"Vector database: ChromaDB")
    print(f"Collection: financial_documents")
    print("="*70)

    # Get collection stats
    try:
        stats = vector_store.get_stats()
        print(f"\nVector Store Stats:")
        print(f"  Total documents: {stats.get('count', 0):,}")
    except Exception as e:
        logger.error(f"Could not get stats: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest chunked documents into vector database"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Ingest a specific chunked file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Ingest all chunked files (default)"
    )

    args = parser.parse_args()

    if args.file:
        logger.info(f"Ingesting single file: {args.file}")
        result = ingest_chunked_file(args.file)
        if result["success"]:
            print(f"✓ Successfully ingested {result['chunks_ingested']} chunks")
        else:
            print(f"✗ Failed: {result.get('error')}")
    else:
        logger.info("Starting chunked documents ingestion to vector database")
        ingest_all_chunks()
        logger.info("Ingestion complete")
