"""
Chunk Documents - Hybrid Chunking Strategy Execution

Main entry point for running the hybrid chunking strategy on parsed documents.

Usage:
    # Test mode - chunk specific files
    python chunk_documents.py --test

    # Single file
    python chunk_documents.py --file data_parsed/AAPL/edgar/AAPL_10K_2024.jsonl

    # Batch process all files
    python chunk_documents.py --all

    # Batch process specific ticker
    python chunk_documents.py --ticker AAPL

    # Dry run (validate without saving)
    python chunk_documents.py --test --dry-run
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
import logging
from collections import Counter
import statistics

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.services.data_ingestion.chunker.runner import (
    load_config,
    chunk_document,
    save_chunked_output
)
from app.services.data_ingestion.chunker.validate import (
    validate_chunks,
    print_validation_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_statistics(chunks: List[Dict]) -> Dict:
    """
    Compute statistics for chunked output.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Statistics dictionary with token distribution, chunk types, etc.
    """
    if not chunks:
        return {}

    # Token statistics
    token_counts = [c.get("chunk_tokens", 0) for c in chunks]
    token_stats = {
        "min": min(token_counts),
        "max": max(token_counts),
        "mean": statistics.mean(token_counts),
        "median": statistics.median(token_counts),
        "total": sum(token_counts)
    }

    # Chunk type distribution
    chunk_types = Counter(c.get("chunk_type", "unknown") for c in chunks)

    # Section/phase coverage
    doc_type = chunks[0].get("doc_type")

    if doc_type in ["10-K", "10-Q"]:
        # EDGAR: count sections
        sections = Counter(
            c.get("section_id", "UNKNOWN")
            for c in chunks
            if c.get("section_id")
        )
        structure_coverage = {
            "type": "sections",
            "breakdown": dict(sections)
        }
    elif doc_type == "earnings_transcript":
        # Transcript: count phases
        phases = Counter(c.get("phase", "unknown") for c in chunks)
        structure_coverage = {
            "type": "phases",
            "breakdown": dict(phases)
        }
    else:
        structure_coverage = {}

    # Overlap statistics
    overlaps_prev = [c.get("overlap_with_prev", 0) for c in chunks]
    overlaps_next = [c.get("overlap_with_next", 0) for c in chunks]

    overlap_stats = {
        "chunks_with_prev_overlap": sum(1 for o in overlaps_prev if o > 0),
        "chunks_with_next_overlap": sum(1 for o in overlaps_next if o > 0),
        "avg_prev_overlap": statistics.mean(overlaps_prev) if overlaps_prev else 0,
        "avg_next_overlap": statistics.mean(overlaps_next) if overlaps_next else 0
    }

    return {
        "total_chunks": len(chunks),
        "doc_type": doc_type,
        "token_stats": token_stats,
        "chunk_types": dict(chunk_types),
        "structure_coverage": structure_coverage,
        "overlap_stats": overlap_stats
    }


def print_statistics(stats: Dict, filename: str):
    """
    Print human-readable statistics.

    Args:
        stats: Statistics dictionary from compute_statistics()
        filename: Source filename
    """
    print("\n" + "="*60)
    print(f"CHUNKING STATISTICS: {filename}")
    print("="*60)

    print(f"\nDocument Type: {stats['doc_type']}")
    print(f"Total Chunks: {stats['total_chunks']}")

    print("\n--- Token Distribution ---")
    ts = stats['token_stats']
    print(f"  Min: {ts['min']} tokens")
    print(f"  Max: {ts['max']} tokens")
    print(f"  Mean: {ts['mean']:.1f} tokens")
    print(f"  Median: {ts['median']:.1f} tokens")
    print(f"  Total: {ts['total']:,} tokens")

    print("\n--- Chunk Types ---")
    for chunk_type, count in sorted(
        stats['chunk_types'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {chunk_type}: {count} ({count/stats['total_chunks']*100:.1f}%)")

    if stats.get('structure_coverage'):
        sc = stats['structure_coverage']
        print(f"\n--- {sc['type'].title()} Coverage ---")
        for key, count in sorted(
            sc['breakdown'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {key}: {count} chunks")

    print("\n--- Overlap Statistics ---")
    os = stats['overlap_stats']
    print(f"  Chunks with prev overlap: {os['chunks_with_prev_overlap']}")
    print(f"  Chunks with next overlap: {os['chunks_with_next_overlap']}")
    print(f"  Avg prev overlap: {os['avg_prev_overlap']:.1f} tokens")
    print(f"  Avg next overlap: {os['avg_next_overlap']:.1f} tokens")

    print("="*60 + "\n")


def chunk_single_file(
    input_path: Path,
    config: Dict,
    output_dir: Path,
    dry_run: bool = False
) -> bool:
    """
    Chunk a single file.

    Args:
        input_path: Path to input JSONL file
        config: Chunking configuration
        output_dir: Output directory for chunks
        dry_run: If True, validate but don't save

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing: {input_path}")

    try:
        # Run chunking
        chunks = chunk_document(input_path, config)

        if not chunks:
            logger.warning(f"No chunks created for {input_path}")
            return False

        # Compute statistics
        stats = compute_statistics(chunks)
        print_statistics(stats, input_path.name)

        # Validate chunks
        report = validate_chunks(
            chunks,
            min_tokens=config.get("min_chunk_tokens", 200),
            max_tokens=config.get("max_chunk_tokens", 800),
            verbose=False
        )
        print_validation_report(report)

        # Save output
        if not dry_run:
            # Determine output path
            # Input: data_parsed/AAPL/edgar/AAPL_10K_2024.jsonl
            # Output: data_chunked/AAPL/edgar/AAPL_10K_2024_chunks.jsonl

            # Extract ticker and doc_type directory
            parts = input_path.parts
            ticker_idx = parts.index("data_parsed") + 1
            ticker = parts[ticker_idx]
            doc_dir = parts[ticker_idx + 1]  # "edgar" or "earnings"

            output_path = output_dir / ticker / doc_dir / f"{input_path.stem}_chunks.jsonl"

            save_chunked_output(chunks, output_path)

            print(f"✓ Saved {len(chunks)} chunks to {output_path}\n")

        return True

    except Exception as e:
        logger.error(f"Failed to chunk {input_path}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Chunk parsed documents using hybrid strategy"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: chunk AAPL 10-K 2024 and earnings 2024 Q4"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Chunk a specific file (path relative to project root)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Batch process all parsed files"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Batch process all files for specific ticker"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without saving output"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="backend/app/services/data_ingestion/configs/chunking.yaml",
        help="Path to chunking config YAML"
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.test, args.file, args.all, args.ticker]):
        parser.error("Must specify one of: --test, --file, --all, --ticker")

    # Paths
    project_root = Path(__file__).parent
    config_path = project_root / args.config
    data_parsed_dir = project_root / "data_parsed"
    output_dir = project_root / "data_chunked"

    # Load config
    logger.info(f"Loading config from {config_path}")
    config = load_config(str(config_path))

    # Determine files to process
    files_to_process = []

    if args.test:
        # Test mode: AAPL samples
        files_to_process = [
            data_parsed_dir / "AAPL" / "edgar" / "AAPL_10K_2024.jsonl",
            data_parsed_dir / "AAPL" / "earnings" / "2024_Q4.jsonl"
        ]

    elif args.file:
        # Single file
        file_path = project_root / args.file
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)
        files_to_process = [file_path]

    elif args.ticker:
        # All files for ticker
        ticker_dir = data_parsed_dir / args.ticker
        if not ticker_dir.exists():
            logger.error(f"Ticker directory not found: {ticker_dir}")
            sys.exit(1)

        # Find all JSONL files
        files_to_process = list(ticker_dir.rglob("*.jsonl"))

    elif args.all:
        # All files
        files_to_process = list(data_parsed_dir.rglob("*.jsonl"))

    # Validate files exist
    files_to_process = [f for f in files_to_process if f.exists()]

    if not files_to_process:
        logger.error("No files found to process")
        sys.exit(1)

    logger.info(f"Found {len(files_to_process)} files to process")

    # Process files
    success_count = 0
    failed_files = []

    for file_path in files_to_process:
        success = chunk_single_file(file_path, config, output_dir, args.dry_run)

        if success:
            success_count += 1
        else:
            failed_files.append(file_path)

    # Summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files: {len(files_to_process)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_files)}")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")

    print("="*60 + "\n")

    if args.dry_run:
        print("DRY RUN: No output files saved\n")

    sys.exit(0 if len(failed_files) == 0 else 1)


if __name__ == "__main__":
    main()
