#!/usr/bin/env python3
"""
Process and chunk financial documents with duplicate prevention.
Creates JSONL chunk files in data/processed/chunks/
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Set
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.modules.chunker import DocumentChunker
except ModuleNotFoundError:
    from modules.chunker import DocumentChunker


def get_processed_documents(output_dir: Path) -> Set[str]:
    """
    Get set of already processed document IDs from existing chunk files.

    Returns:
        Set of doc_ids that have already been chunked
    """
    processed = set()

    if not output_dir.exists():
        return processed

    # Scan all existing chunk JSONL files
    for chunk_file in output_dir.glob("**/*_chunks.jsonl"):
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                    doc_id = chunk.get('doc_id', '')
                    if doc_id:
                        processed.add(doc_id)
                except json.JSONDecodeError:
                    pass

    return processed


def chunk_document(
    filepath: Path,
    chunker: DocumentChunker,
    doc_id: str,
    metadata: Dict
) -> List[Dict]:
    """Chunk a single document"""

    # Read document content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content.strip():
        print(f"  ⚠️  Empty file, skipping")
        return []

    # Chunk using DocumentChunker
    chunks = chunker.chunk(content, metadata)

    # Convert to JSONL format
    chunk_dicts = []
    for chunk in chunks:
        chunk_dict = {
            "chunk_id": chunk.chunk_id,
            "doc_id": doc_id,
            "text": chunk.text,
            "chunk_tokens": len(chunker.tokenizer.encode(chunk.text)),
            "start_idx": chunk.start_idx,
            "end_idx": chunk.end_idx,
            **chunk.metadata
        }
        chunk_dicts.append(chunk_dict)

    return chunk_dicts


def save_chunks(chunks: List[Dict], output_file: Path):
    """Save chunks to JSONL file"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write mode (overwrite if exists to avoid duplicates)
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')


def extract_metadata_from_path(filepath: Path) -> Dict:
    """Extract metadata from file path"""
    import re

    metadata = {
        "filename": filepath.name,
        "source": "unknown"
    }

    # Extract company from path
    parts = filepath.parts
    if 'edgar' in parts or 'earnings' in parts:
        # Find company ticker in path
        for part in parts:
            if part.isupper() and len(part) <= 5:
                metadata["company"] = part
                break

    # Extract doc_type
    filename_lower = filepath.name.lower()
    if '10-k' in filename_lower or '10k' in filename_lower:
        metadata["doc_type"] = "10-K"
    elif '10-q' in filename_lower or '10q' in filename_lower:
        metadata["doc_type"] = "10-Q"
    elif 'earnings' in filename_lower or 'transcript' in filename_lower:
        metadata["doc_type"] = "earnings_transcript"
    else:
        metadata["doc_type"] = "other"

    # Extract year/quarter
    year_match = re.search(r'20\d{2}', filepath.name)
    if year_match:
        metadata["fiscal_year"] = int(year_match.group())

    quarter_match = re.search(r'[Qq]([1-4])', filepath.name)
    if quarter_match:
        metadata["quarter"] = f"Q{quarter_match.group(1)}"

    # Source
    if 'edgar' in str(filepath):
        metadata["source"] = "edgar"
    elif 'earnings' in str(filepath):
        metadata["source"] = "earnings"

    return metadata


def process_documents(
    input_dir: Path,
    output_dir: Path,
    skip_existing: bool = True,
    chunker_strategy: str = "window",
    chunk_size: int = 512,
    chunk_overlap: int = 128
):
    """
    Process documents with duplicate prevention.

    Args:
        input_dir: Directory containing source documents
        output_dir: Directory to save chunk JSONL files
        skip_existing: Skip documents that have already been chunked
        chunker_strategy: Chunking strategy ('window', 'semantic', 'section')
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
    """

    # Get already processed documents
    processed_docs = get_processed_documents(output_dir) if skip_existing else set()

    if processed_docs:
        print(f"📊 Found {len(processed_docs)} already processed documents")

    # Initialize chunker
    chunker = DocumentChunker(
        strategy=chunker_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Find all source documents
    source_files = list(input_dir.glob("**/*.txt")) + list(input_dir.glob("**/*.jsonl"))

    if not source_files:
        print(f"❌ No documents found in {input_dir}")
        return

    print(f"\n📁 Found {len(source_files)} source documents")
    print(f"🔄 Processing with '{chunker_strategy}' strategy")
    print(f"   Chunk size: {chunk_size} tokens, Overlap: {chunk_overlap} tokens\n")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for filepath in source_files:
        # Generate doc_id from filename
        doc_id = filepath.stem

        # Extract metadata
        metadata = extract_metadata_from_path(filepath)
        metadata["doc_id"] = doc_id

        # Skip if already processed
        if doc_id in processed_docs:
            skipped_count += 1
            print(f"⏭️  Skipping {doc_id} (already processed)")
            continue

        print(f"📄 Processing: {doc_id}")

        try:
            # Chunk the document
            chunks = chunk_document(filepath, chunker, doc_id, metadata)

            if not chunks:
                print(f"  ⚠️  No chunks generated")
                continue

            # Determine output path
            company = metadata.get("company", "UNKNOWN")
            source = metadata.get("source", "other")
            output_file = output_dir / company / source / f"{doc_id}_chunks.jsonl"

            # Save chunks
            save_chunks(chunks, output_file)

            print(f"  ✓ Created {len(chunks)} chunks → {output_file}")
            processed_count += 1

        except Exception as e:
            print(f"  ❌ Error: {e}")
            error_count += 1

    # Summary
    print(f"\n{'='*60}")
    print("Processing Summary:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped:   {skipped_count} (already existed)")
    print(f"  Errors:    {error_count}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Process and chunk financial documents with duplicate prevention"
    )
    parser.add_argument("--input-dir", default="data/raw",
                       help="Input directory containing source documents")
    parser.add_argument("--output-dir", default="data/processed/chunks",
                       help="Output directory for chunk JSONL files")
    parser.add_argument("--force", action="store_true",
                       help="Re-process all documents (ignore existing chunks)")
    parser.add_argument("--strategy", default="window",
                       choices=["window", "semantic", "section", "sentence"],
                       help="Chunking strategy")
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=128,
                       help="Chunk overlap in tokens")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1

    process_documents(
        input_dir=input_dir,
        output_dir=output_dir,
        skip_existing=not args.force,
        chunker_strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    return 0


if __name__ == "__main__":
    exit(main())
