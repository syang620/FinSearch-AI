#!/usr/bin/env python3
"""
Unified CLI for embedding operations.
Supports multiple providers and handles the complete pipeline from chunks to ChromaDB.

Usage examples:
    # Generate embeddings with Ollama
    python embed_cli.py generate --provider ollama --model nomic-embed-text

    # Generate embeddings with FinE5 API
    python embed_cli.py generate --provider fine5 --archive chunks_for_colab.tar.gz

    # Ingest embeddings into ChromaDB
    python embed_cli.py ingest --input embeddings.jsonl --chroma-path data/chroma_db

    # Complete pipeline
    python embed_cli.py pipeline --provider fine5 --archive chunks.tar.gz --chroma-path data/chroma_db
"""

import os
import sys
import json
import tarfile
import argparse
import shutil
from pathlib import Path
from typing import List, Dict
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try both import styles for flexibility
try:
    # When run from project root: python scripts/embed_cli.py
    from scripts.modules.embedder import (
        create_embedder,
        process_chunks_with_embeddings,
        save_embeddings_to_jsonl,
        load_embeddings_from_jsonl,
        EmbeddingResult
    )
    from scripts.modules.chunker import DocumentChunker
except ModuleNotFoundError:
    # When run from scripts directory: python embed_cli.py
    from modules.embedder import (
        create_embedder,
        process_chunks_with_embeddings,
        save_embeddings_to_jsonl,
        load_embeddings_from_jsonl,
        EmbeddingResult
    )
    from modules.chunker import DocumentChunker
import chromadb
from tqdm import tqdm


# ========= Chunk Extraction =========

def extract_archive(archive_path: str, extract_dir: str = "extracted_chunks") -> str:
    """Extract tar.gz archive"""
    Path(extract_dir).mkdir(parents=True, exist_ok=True)

    print(f"Extracting {archive_path} to {extract_dir}...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(extract_dir)

    return extract_dir


def collect_chunks_from_directory(data_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """Collect all chunks from directory - supports both JSONL and text files"""
    chunks = []
    data_path = Path(data_path)

    # First, try to find JSONL files (pre-chunked data)
    jsonl_files = list(data_path.glob("**/*.jsonl"))

    if jsonl_files:
        print(f"Found {len(jsonl_files)} JSONL chunk files")
        for filepath in jsonl_files:
            if filepath.name.startswith('.'):
                continue

            print(f"  Loading chunks from: {filepath.name}")

            # Read JSONL file
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        chunk_data = json.loads(line)

                        # Extract metadata from filepath for additional context
                        file_metadata = extract_metadata_from_filepath(filepath)

                        # Combine metadata
                        metadata = {
                            **file_metadata,
                            "doc_id": chunk_data.get("doc_id", ""),
                            "chunk_index": chunks.__len__()  # Track order
                        }

                        # Add chunk with its text and metadata
                        chunks.append({
                            "chunk_id": chunk_data.get("chunk_id", f"chunk_{len(chunks)}"),
                            "text": chunk_data.get("text", ""),
                            "metadata": metadata
                        })
                    except json.JSONDecodeError:
                        print(f"    Warning: Skipping invalid JSON line")

    # If no JSONL files, fall back to text files
    else:
        txt_files = list(data_path.glob("**/*.txt"))
        if txt_files:
            print(f"Found {len(txt_files)} text files, chunking them...")

            chunker = DocumentChunker(
                strategy="window",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            for filepath in txt_files:
                if filepath.name.startswith('.'):
                    continue

                print(f"  Processing: {filepath.name}")

                # Read content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract metadata
                metadata = extract_metadata_from_filepath(filepath)

                # Chunk the document
                doc_chunks = chunker.chunk_document(content, metadata)

                # Convert to expected format
                for idx, chunk in enumerate(doc_chunks):
                    chunk_id = f"{metadata['company']}:{metadata['document_type']}:{metadata.get('year', 'unknown')}:{idx}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk.text,
                        "metadata": {**metadata, **chunk.metadata}
                    })

    return chunks


def extract_metadata_from_filepath(filepath: Path) -> Dict:
    """Extract metadata from file path"""
    filename = filepath.name
    full_path = str(filepath)

    metadata = {
        "filename": filename,
        "filepath": str(filepath),
        "file_type": filepath.suffix,
        "source": "edgar"
    }

    # Extract company
    companies = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA', 'UNH', 'V', 'AVGO']
    company = None

    for part in filepath.parts:
        if part.upper() in companies:
            company = part.upper()
            break

    if not company:
        company_match = re.search(r'^([A-Z]+)_', filename)
        if company_match:
            company = company_match.group(1)

    metadata["company"] = company or "UNKNOWN"

    # Extract document type
    if "10-K" in filename or "10-k" in filename.lower():
        metadata["document_type"] = "10-K"
    elif "10-Q" in filename or "10-q" in filename.lower():
        metadata["document_type"] = "10-Q"
    elif "8-K" in filename or "8-k" in filename.lower():
        metadata["document_type"] = "8-K"
    else:
        metadata["document_type"] = "other"

    # Extract year
    year_match = re.search(r'20\d{2}', filename)
    if year_match:
        metadata["year"] = int(year_match.group())

    return metadata


# ========= ChromaDB Ingestion =========

def ingest_to_chromadb(
    embeddings: List[EmbeddingResult],
    chroma_path: str,
    collection_name: str = "financial_documents",
    batch_size: int = 100,
    reset: bool = False
) -> None:
    """Ingest embeddings into ChromaDB"""

    # Clear existing database if requested
    if reset and os.path.exists(chroma_path):
        print(f"Removing existing database at {chroma_path}")
        shutil.rmtree(chroma_path)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_path)

    # Create or get collection
    try:
        if reset:
            client.delete_collection(collection_name)
    except:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"\nIngesting {len(embeddings)} embeddings into ChromaDB...")

    # Process in batches
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Ingesting"):
        batch = embeddings[i:i + batch_size]

        ids = [e.id for e in batch]
        texts = [e.text for e in batch]
        embeddings_list = [e.embedding for e in batch]
        metadatas = [e.metadata for e in batch]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas
        )

    print(f"✓ Ingested {collection.count()} documents into ChromaDB")
    print(f"✓ Database ready at: {chroma_path}")


# ========= Command Functions =========

def cmd_generate(args):
    """Generate embeddings command"""
    # Determine input source
    if args.archive:
        # Extract and collect chunks from archive
        extract_dir = extract_archive(args.archive)
        chunks = collect_chunks_from_directory(extract_dir, args.chunk_size, args.chunk_overlap)
    elif args.data_path:
        # Collect chunks from directory
        chunks = collect_chunks_from_directory(args.data_path, args.chunk_size, args.chunk_overlap)
    else:
        print("Error: Specify either --archive or --data-path")
        return 1

    print(f"\nCollected {len(chunks)} chunks")

    # Create embedder
    embedder_kwargs = {}
    if args.provider == "ollama":
        embedder_kwargs["model_name"] = args.model
    elif args.provider == "fine5":
        embedder_kwargs["api_key"] = args.api_key
        if hasattr(args, 'no_instruction') and args.no_instruction:
            embedder_kwargs["use_instruction"] = False
        if hasattr(args, 'task_description') and args.task_description:
            embedder_kwargs["task_description"] = args.task_description
    elif args.provider in ["sentence-transformers", "st"]:
        embedder_kwargs["model_name"] = args.model

    embedder = create_embedder(args.provider, **embedder_kwargs)

    # Process chunks
    results = process_chunks_with_embeddings(chunks, embedder, args.batch_size)

    # Save results
    save_embeddings_to_jsonl(results, args.output)

    return 0


def cmd_ingest(args):
    """Ingest embeddings into ChromaDB"""
    # Load embeddings
    embeddings = load_embeddings_from_jsonl(args.input)

    # Ingest to ChromaDB
    ingest_to_chromadb(
        embeddings,
        args.chroma_path,
        args.collection_name,
        args.batch_size,
        args.reset
    )

    return 0


def cmd_pipeline(args):
    """Run complete pipeline: extract -> embed -> ingest"""
    # Generate embeddings
    print("=" * 60)
    print("Step 1: Generating Embeddings")
    print("=" * 60)

    if cmd_generate(args) != 0:
        return 1

    # Ingest to ChromaDB
    print("\n" + "=" * 60)
    print("Step 2: Ingesting to ChromaDB")
    print("=" * 60)

    args.input = args.output  # Use generated embeddings
    if cmd_ingest(args) != 0:
        return 1

    print("\n" + "=" * 60)
    print("✓ Pipeline Complete!")
    print("=" * 60)

    return 0


# ========= Main CLI =========

def main():
    parser = argparse.ArgumentParser(description="Unified embedding CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate embeddings")
    gen_parser.add_argument("--provider", default="ollama",
                           choices=["ollama", "fine5", "sentence-transformers", "st", "nv-embed-v2", "nvembed"],
                           help="Embedding provider")
    gen_parser.add_argument("--model", default="nomic-embed-text",
                           help="Model name (for ollama/sentence-transformers)")
    gen_parser.add_argument("--api-key", help="API key (for fine5)")
    gen_parser.add_argument("--archive", help="Input tar.gz archive")
    gen_parser.add_argument("--data-path", help="Input data directory")
    gen_parser.add_argument("--output", default="embeddings.jsonl",
                           help="Output JSONL file")
    gen_parser.add_argument("--batch-size", type=int, default=16,
                           help="Batch size for embedding")
    gen_parser.add_argument("--chunk-size", type=int, default=1000,
                           help="Chunk size in characters")
    gen_parser.add_argument("--chunk-overlap", type=int, default=200,
                           help="Chunk overlap in characters")
    gen_parser.add_argument("--no-instruction", action="store_true",
                           help="Disable instruction prompting (for fine5, use raw text)")
    gen_parser.add_argument("--task-description", type=str,
                           help="Custom task description for instruction prompting (fine5 only)")

    # Ingest command
    ing_parser = subparsers.add_parser("ingest", help="Ingest embeddings to ChromaDB")
    ing_parser.add_argument("--input", default="embeddings.jsonl",
                           help="Input JSONL file")
    ing_parser.add_argument("--chroma-path", default="data/chroma_db",
                           help="ChromaDB path")
    ing_parser.add_argument("--collection-name", default="financial_documents",
                           help="Collection name")
    ing_parser.add_argument("--batch-size", type=int, default=100,
                           help="Batch size for ingestion")
    ing_parser.add_argument("--reset", action="store_true",
                           help="Reset database before ingestion")

    # Pipeline command (combines generate + ingest)
    pip_parser = subparsers.add_parser("pipeline", help="Run complete pipeline")
    pip_parser.add_argument("--provider", default="ollama",
                           choices=["ollama", "fine5", "sentence-transformers", "st", "nv-embed-v2", "nvembed"],
                           help="Embedding provider")
    pip_parser.add_argument("--model", default="nomic-embed-text",
                           help="Model name")
    pip_parser.add_argument("--api-key", help="API key (for fine5)")
    pip_parser.add_argument("--archive", help="Input tar.gz archive")
    pip_parser.add_argument("--data-path", help="Input data directory")
    pip_parser.add_argument("--output", default="embeddings.jsonl",
                           help="Embeddings file")
    pip_parser.add_argument("--chroma-path", default="data/chroma_db",
                           help="ChromaDB path")
    pip_parser.add_argument("--collection-name", default="financial_documents",
                           help="Collection name")
    pip_parser.add_argument("--batch-size", type=int, default=16,
                           help="Batch size")
    pip_parser.add_argument("--chunk-size", type=int, default=1000,
                           help="Chunk size")
    pip_parser.add_argument("--chunk-overlap", type=int, default=200,
                           help="Chunk overlap")
    pip_parser.add_argument("--reset", action="store_true",
                           help="Reset database")
    pip_parser.add_argument("--no-instruction", action="store_true",
                           help="Disable instruction prompting (for fine5, use raw text)")
    pip_parser.add_argument("--task-description", type=str,
                           help="Custom task description for instruction prompting (fine5 only)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Set API key from environment if not provided
    if args.provider == "fine5" and not args.api_key:
        args.api_key = os.environ.get("ABACI_API_KEY")
        if not args.api_key:
            print("Error: ABACI_API_KEY not set. Use --api-key or export ABACI_API_KEY")
            return 1

    # Execute command
    commands = {
        "generate": cmd_generate,
        "ingest": cmd_ingest,
        "pipeline": cmd_pipeline
    }

    return commands[args.command](args)


if __name__ == "__main__":
    exit(main())