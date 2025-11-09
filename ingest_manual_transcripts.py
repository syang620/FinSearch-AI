#!/usr/bin/env python3
"""
Manual Earnings Call Transcripts Ingestion

Processes manually downloaded PDF transcripts and saves them to the system.
"""

import sys
import os
from pathlib import Path
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.data_ingestion.earnings.pdf_parser import pdf_parser
from app.services.data_ingestion.file_storage import file_storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_pdf_files(base_dir: str) -> list:
    """
    Find all PDF files in the manual transcripts directory

    Args:
        base_dir: Base directory to search

    Returns:
        List of PDF file paths
    """
    base_path = Path(base_dir)
    pdf_files = list(base_path.glob('**/*.pdf'))

    logger.info(f"Found {len(pdf_files)} PDF files in {base_dir}")
    return sorted(pdf_files)


def ingest_transcript(pdf_path: Path) -> bool:
    """
    Ingest a single transcript PDF

    Args:
        pdf_path: Path to PDF file

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing: {pdf_path.name}")

    try:
        # Parse PDF
        transcript_data = pdf_parser.parse_pdf(str(pdf_path))

        if not transcript_data:
            logger.error(f"Failed to parse PDF: {pdf_path.name}")
            return False

        # Validate data
        ticker = transcript_data['ticker']
        year = transcript_data['year']
        quarter = transcript_data['quarter']
        transcript_text = transcript_data['transcript']

        if not transcript_text or len(transcript_text) < 100:
            logger.error(f"Transcript too short ({len(transcript_text)} chars): {pdf_path.name}")
            return False

        # Save to file storage
        file_storage.save_earnings_call(
            ticker=ticker,
            year=year,
            quarter=quarter,
            transcript=transcript_text,
            metadata={
                'source': 'manual_pdf',
                'source_file': transcript_data['source_file'],
                'source_path': transcript_data['source_path'],
                'file_size': transcript_data['file_size'],
                'character_count': len(transcript_text),
                'word_count': len(transcript_text.split())
            }
        )

        logger.info(
            f"✓ {ticker} Q{quarter} {year}: "
            f"{len(transcript_text):,} chars, "
            f"{len(transcript_text.split()):,} words"
        )

        return True

    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main ingestion flow"""

    print("=" * 70)
    print("Manual Earnings Call Transcripts Ingestion")
    print("=" * 70)
    print()

    # Directory containing manual PDFs
    manual_dir = Path(__file__).parent / "data" / "earnings_calls_manual"

    if not manual_dir.exists():
        print(f"ERROR: Manual transcripts directory not found: {manual_dir}")
        print()
        print("Please ensure PDFs are saved to:")
        print(f"  {manual_dir}")
        print()
        return 1

    # Find all PDF files
    pdf_files = find_pdf_files(manual_dir)

    if not pdf_files:
        print("ERROR: No PDF files found")
        print()
        print(f"Please add PDF files to: {manual_dir}")
        print()
        return 1

    print(f"Found {len(pdf_files)} PDF files")
    print()

    # Group by company
    by_company = {}
    for pdf_file in pdf_files:
        company = pdf_file.parent.name
        if company not in by_company:
            by_company[company] = []
        by_company[company].append(pdf_file)

    print(f"Companies: {len(by_company)}")
    for company, files in sorted(by_company.items()):
        print(f"  • {company}: {len(files)} transcripts")
    print()

    print("-" * 70)
    print()

    # Process each PDF
    successful = 0
    failed = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_file.parent.name}/{pdf_file.name}")

        if ingest_transcript(pdf_file):
            successful += 1
        else:
            failed += 1

        print()

    # Summary
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print()
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    if failed > 0:
        print(f"⚠ {failed} files failed to process - check logs above")
        print()

    # Show where data was saved
    print("Data saved to:")
    print(f"  {file_storage.earnings_dir}")
    print()

    # Show stats by company
    print("=" * 70)
    print("TRANSCRIPTS BY COMPANY")
    print("=" * 70)

    for company in sorted(by_company.keys()):
        company_dir = file_storage.earnings_dir / company
        if company_dir.exists():
            transcripts = list(company_dir.glob('*.txt'))
            print(f"{company:8s}: {len(transcripts)} transcripts")

    print()

    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Inspect transcript files:")
    print(f"   cd {file_storage.earnings_dir}")
    print(f"   cat AAPL/2024_Q4.txt")
    print()
    print("2. Run RAG ingestion to load into vector database:")
    print(f"   python ingest_to_rag.py")
    print()
    print("3. Test queries via chat interface")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
