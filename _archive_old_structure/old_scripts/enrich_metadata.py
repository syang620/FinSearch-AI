#!/usr/bin/env python3
"""
Metadata Enrichment Script

Enriches existing JSONL files with missing metadata fields to conform to
the unified metadata schema.

Adds:
- chunk_id
- company (set to ticker)
- fiscal_year
- quarter
- period
- source_file
- parsed_at (current timestamp for re-enrichment)
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import logging
import re

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.data_ingestion.metadata_schema import (
    compute_period,
    compute_chunk_id,
    get_current_timestamp
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_quarter_from_filename(filename: str, doc_type: str) -> str:
    """
    Extract quarter from filename

    Args:
        filename: JSONL filename
        doc_type: Document type

    Returns:
        Quarter string (Q1-Q4, FY)
    """
    if doc_type == '10-K':
        return 'FY'

    # Try to extract quarter from filename
    # Format: TICKER_10Q_YEAR_Q3.jsonl or TICKER_TRANSCRIPT_YEAR_Q4.jsonl
    match = re.search(r'_Q(\d)', filename)
    if match:
        return f"Q{match.group(1)}"

    # Default to Q1 if not found
    return 'Q1'


def enrich_edgar_metadata(jsonl_path: Path) -> int:
    """
    Enrich EDGAR JSONL file with missing metadata

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        Number of chunks enriched
    """
    logger.info(f"Enriching EDGAR file: {jsonl_path}")

    # Read all chunks
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        logger.warning(f"No chunks found in {jsonl_path}")
        return 0

    # Get metadata from first chunk
    sample = chunks[0]
    ticker = sample.get('ticker')
    doc_type = sample.get('doc_type')
    filing_date = sample.get('filing_date')
    doc_id = sample.get('doc_id')

    # Extract fiscal year from filing_date or doc_id
    if filing_date:
        fiscal_year = int(filing_date[:4])
    elif doc_id:
        # Try to extract year from doc_id (e.g., AAPL_10K_2024)
        match = re.search(r'_(\d{4})', doc_id)
        fiscal_year = int(match.group(1)) if match else 2024
    else:
        fiscal_year = 2024  # Default

    # Determine quarter
    quarter = extract_quarter_from_filename(jsonl_path.name, doc_type)

    # Compute standardized fields
    period = compute_period(fiscal_year, quarter)
    parsed_at = get_current_timestamp()

    # Determine source file path
    # Map from data_parsed back to data
    ticker_from_path = jsonl_path.parent.parent.name
    if doc_type == '10-K':
        source_file = f"data/edgar/{ticker_from_path}/raw_10-K_{fiscal_year}.htm"
    elif doc_type == '10-Q':
        q_num = quarter[1] if quarter.startswith('Q') else '1'
        source_file = f"data/edgar/{ticker_from_path}/raw_10-Q_{fiscal_year}_{quarter}.htm"
    else:
        source_file = f"data/edgar/{ticker_from_path}/raw_{doc_type}_{fiscal_year}.htm"

    # Enrich each chunk
    enriched_chunks = []
    for idx, chunk in enumerate(chunks):
        # Add missing fields
        if 'chunk_id' not in chunk:
            chunk['chunk_id'] = compute_chunk_id(
                chunk.get('doc_id', doc_id),
                chunk.get('unit_index', idx),
                chunk.get('unit_type', 'paragraph')
            )

        if 'company' not in chunk:
            chunk['company'] = chunk.get('ticker', ticker)

        if 'fiscal_year' not in chunk:
            chunk['fiscal_year'] = fiscal_year

        if 'quarter' not in chunk:
            chunk['quarter'] = quarter

        if 'period' not in chunk:
            chunk['period'] = period

        if 'source_file' not in chunk:
            chunk['source_file'] = source_file

        if 'parsed_at' not in chunk:
            chunk['parsed_at'] = parsed_at

        # Ensure transcript-specific fields are null for EDGAR files
        transcript_fields = [
            'phase', 'speaker_name', 'speaker_role', 'speaker_firm',
            'utterance_id', 'utterance_type', 'token_count',
            'exchange_id', 'exchange_role'
        ]
        for field in transcript_fields:
            if field not in chunk:
                chunk[field] = None

        enriched_chunks.append(chunk)

    # Write back to file
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for chunk in enriched_chunks:
            f.write(json.dumps(chunk) + '\n')

    logger.info(f"  Enriched {len(enriched_chunks)} chunks")
    return len(enriched_chunks)


def enrich_transcript_metadata(jsonl_path: Path) -> int:
    """
    Enrich transcript JSONL file with missing metadata

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        Number of chunks enriched
    """
    logger.info(f"Enriching transcript file: {jsonl_path}")

    # Read all chunks
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        logger.warning(f"No chunks found in {jsonl_path}")
        return 0

    # Get metadata from first chunk or filename
    sample = chunks[0]
    ticker = sample.get('ticker')

    # Extract year and quarter from filename: YYYY_QN.jsonl
    filename = jsonl_path.stem
    match = re.match(r'(\d{4})_Q(\d)', filename)
    if match:
        fiscal_year = int(match.group(1))
        quarter_num = int(match.group(2))
        quarter = f"Q{quarter_num}"
    else:
        # Fall back to chunk data
        fiscal_year = sample.get('year', sample.get('fiscal_year', 2024))
        quarter_num = sample.get('quarter', 1)
        if isinstance(quarter_num, int):
            quarter = f"Q{quarter_num}"
        else:
            quarter = quarter_num

    # Compute standardized fields
    period = compute_period(fiscal_year, quarter)
    parsed_at = get_current_timestamp()
    filing_date = f"{fiscal_year}-{int(quarter[1]) * 3:02d}-01"  # Approximate

    # Determine source file path
    ticker_from_path = jsonl_path.parent.parent.name
    source_file = f"data/earnings_calls_manual/{ticker_from_path}/{ticker_from_path}_FY{fiscal_year}_{quarter}.pdf"

    # Determine doc_id
    doc_id = f"{ticker}_TRANSCRIPT_{fiscal_year}_{quarter}"

    # Enrich each chunk
    enriched_chunks = []
    for idx, chunk in enumerate(chunks):
        # Update doc_id format if old format
        if 'doc_id' in chunk and '_TRANSCRIPT_' not in chunk['doc_id']:
            chunk['doc_id'] = doc_id

        # Add missing fields
        if 'chunk_id' not in chunk:
            chunk['chunk_id'] = compute_chunk_id(
                chunk.get('doc_id', doc_id),
                chunk.get('unit_index', idx),
                chunk.get('unit_type', 'utterance')
            )

        if 'company' not in chunk:
            chunk['company'] = chunk.get('ticker', ticker)

        if 'doc_type' not in chunk:
            chunk['doc_type'] = 'earnings_transcript'

        if 'fiscal_year' not in chunk:
            # Migrate from 'year' field if exists
            chunk['fiscal_year'] = chunk.get('year', fiscal_year)

        # Always ensure quarter is string format (Q1-Q4)
        old_quarter = chunk.get('quarter')
        if isinstance(old_quarter, int):
            chunk['quarter'] = f"Q{old_quarter}"
        elif old_quarter is None or not isinstance(old_quarter, str) or not old_quarter.startswith('Q'):
            chunk['quarter'] = quarter

        if 'period' not in chunk:
            chunk['period'] = period

        if 'filing_date' not in chunk:
            chunk['filing_date'] = filing_date

        if 'source_file' not in chunk:
            chunk['source_file'] = source_file

        if 'parsed_at' not in chunk:
            chunk['parsed_at'] = parsed_at

        # Ensure EDGAR-specific fields are null for transcripts
        if 'section_id' not in chunk:
            chunk['section_id'] = None
        if 'section_title' not in chunk:
            chunk['section_title'] = None

        # Ensure speaker_role has a value (default to 'unknown')
        if chunk.get('speaker_role') is None or chunk.get('speaker_role') == 'Unknown':
            chunk['speaker_role'] = 'unknown'

        # Remove old 'year' field (replaced by 'fiscal_year')
        if 'year' in chunk:
            del chunk['year']

        enriched_chunks.append(chunk)

    # Write back to file
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for chunk in enriched_chunks:
            f.write(json.dumps(chunk) + '\n')

    logger.info(f"  Enriched {len(enriched_chunks)} chunks")
    return len(enriched_chunks)


def main():
    """Run metadata enrichment on all JSONL files"""
    start_time = datetime.now()

    print()
    print("=" * 70)
    print("METADATA ENRICHMENT")
    print("=" * 70)
    print()
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    parsed_dir = Path("data_parsed")

    if not parsed_dir.exists():
        print(f"ERROR: {parsed_dir} not found")
        return 1

    # Find all JSONL files
    edgar_files = list(parsed_dir.glob("*/edgar/*.jsonl"))
    transcript_files = list(parsed_dir.glob("*/earnings/*.jsonl"))

    total_files = len(edgar_files) + len(transcript_files)

    print(f"Found {len(edgar_files)} EDGAR files and {len(transcript_files)} transcript files")
    print()

    # Enrich EDGAR files
    print("=" * 70)
    print("ENRICHING EDGAR FILES")
    print("=" * 70)
    print()

    edgar_chunks = 0
    for jsonl_file in edgar_files:
        try:
            chunks = enrich_edgar_metadata(jsonl_file)
            edgar_chunks += chunks
        except Exception as e:
            logger.error(f"Failed to enrich {jsonl_file}: {e}")

    print()

    # Enrich transcript files
    print("=" * 70)
    print("ENRICHING TRANSCRIPT FILES")
    print("=" * 70)
    print()

    transcript_chunks = 0
    for jsonl_file in transcript_files:
        try:
            chunks = enrich_transcript_metadata(jsonl_file)
            transcript_chunks += chunks
        except Exception as e:
            logger.error(f"Failed to enrich {jsonl_file}: {e}")

    print()

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("=" * 70)
    print("ENRICHMENT COMPLETE")
    print("=" * 70)
    print()
    print(f"Files enriched: {total_files}")
    print(f"  EDGAR files: {len(edgar_files)} ({edgar_chunks} chunks)")
    print(f"  Transcript files: {len(transcript_files)} ({transcript_chunks} chunks)")
    print()
    print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
