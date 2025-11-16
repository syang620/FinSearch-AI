#!/usr/bin/env python3
"""
Full Migration Script - Parse All Documents to JSONL Format

Processes all raw documents to normalized JSONL format:
- All EDGAR filings (10-K and 10-Q) from raw HTML
- All earnings call transcripts from PDF
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.data_ingestion.edgar.edgar_parser_v2 import edgar_parser_v2
from app.services.data_ingestion.earnings.transcript_parser_v2 import transcript_parser_v2
from app.services.data_ingestion.file_storage import file_storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_edgar_filings():
    """Migrate all EDGAR filings to JSONL format"""
    print("=" * 70)
    print("MIGRATING EDGAR FILINGS")
    print("=" * 70)
    print()

    edgar_base = Path("data/edgar")
    if not edgar_base.exists():
        print("ERROR: EDGAR directory not found")
        return 0, 0

    # Find all raw HTML files
    raw_files = list(edgar_base.glob("*/raw_*.htm"))
    total = len(raw_files)

    if total == 0:
        print("No EDGAR files found to migrate")
        return 0, 0

    print(f"Found {total} EDGAR filings to process")
    print()

    successful = 0
    failed = 0

    for i, raw_file in enumerate(raw_files, 1):
        ticker = raw_file.parent.name
        filename = raw_file.name

        # Parse filename to get metadata
        # Format: raw_10-K_2025.htm or raw_10-Q_2025_Q3.htm
        try:
            parts = filename.replace('raw_', '').replace('.htm', '').split('_')
            doc_type = parts[0]  # 10-K or 10-Q
            year = parts[1]

            if doc_type == '10-Q' and len(parts) >= 3:
                quarter_str = parts[2]  # Q3
                filing_date = f"{year}-{int(quarter_str[1]) * 3:02d}-01"
            else:
                filing_date = f"{year}-10-31"  # Approximate for 10-K

            print(f"[{i}/{total}] {ticker} {doc_type} {year}")

            # Parse filing
            paragraphs, tables, markdown = edgar_parser_v2.parse_filing_to_jsonl(
                html_path=str(raw_file),
                ticker=ticker,
                doc_type=doc_type,
                filing_date=filing_date
            )

            # Save parsed files
            doc_id = f"{ticker}_{doc_type.replace('-', '')}_{year}"
            if doc_type == '10-Q' and len(parts) >= 3:
                doc_id += f"_Q{quarter_str[1]}"

            file_storage.save_parsed_edgar(
                ticker=ticker,
                doc_id=doc_id,
                paragraphs=paragraphs,
                tables=tables,
                markdown_text=markdown
            )

            print(f"  ✓ {len(paragraphs)} paragraphs, {len(tables)} tables")
            successful += 1

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            logger.error(f"Failed to process {raw_file}: {e}")
            failed += 1

        print()

    return successful, failed


def migrate_earnings_transcripts():
    """Migrate all earnings call transcripts to JSONL format"""
    print("=" * 70)
    print("MIGRATING EARNINGS CALL TRANSCRIPTS")
    print("=" * 70)
    print()

    earnings_base = Path("data/earnings_calls_manual")
    if not earnings_base.exists():
        print("ERROR: Earnings calls manual directory not found")
        return 0, 0

    # Find all PDF files
    pdf_files = list(earnings_base.glob("*/*.pdf"))
    total = len(pdf_files)

    if total == 0:
        print("No earnings transcripts found to migrate")
        return 0, 0

    print(f"Found {total} earnings call transcripts to process")
    print()

    successful = 0
    failed = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        ticker = pdf_file.parent.name
        filename = pdf_file.name

        # Parse filename to get metadata
        # Format: AAPL_FY2025_Q4.pdf
        try:
            parts = filename.replace('.pdf', '').split('_')
            year_str = parts[1].replace('FY', '')
            quarter_str = parts[2].replace('Q', '')

            year = int(year_str)
            quarter = int(quarter_str)

            print(f"[{i}/{total}] {ticker} {year} Q{quarter}")

            # Parse transcript
            utterances, markdown = transcript_parser_v2.parse_transcript_to_jsonl(
                pdf_path=str(pdf_file),
                ticker=ticker,
                year=year,
                quarter=quarter
            )

            # Save parsed files
            file_storage.save_parsed_transcript(
                ticker=ticker,
                year=year,
                quarter=quarter,
                utterances=utterances,
                markdown_text=markdown
            )

            # Count exchanges
            exchanges = len(set(u.get('exchange_id') for u in utterances if u.get('exchange_id')))

            print(f"  ✓ {len(utterances)} utterances, {exchanges} Q&A exchanges")
            successful += 1

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            logger.error(f"Failed to process {pdf_file}: {e}")
            failed += 1

        print()

    return successful, failed


def generate_migration_report(edgar_stats, transcript_stats):
    """Generate migration summary report"""
    print("=" * 70)
    print("MIGRATION COMPLETE")
    print("=" * 70)
    print()

    edgar_success, edgar_failed = edgar_stats
    transcript_success, transcript_failed = transcript_stats

    print("EDGAR FILINGS:")
    print(f"  Successful: {edgar_success}")
    print(f"  Failed: {edgar_failed}")
    print(f"  Total: {edgar_success + edgar_failed}")
    print()

    print("EARNINGS TRANSCRIPTS:")
    print(f"  Successful: {transcript_success}")
    print(f"  Failed: {transcript_failed}")
    print(f"  Total: {transcript_success + transcript_failed}")
    print()

    total_success = edgar_success + transcript_success
    total_failed = edgar_failed + transcript_failed
    total = total_success + total_failed

    print("OVERALL:")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"  Total: {total}")
    print(f"  Success Rate: {(total_success / total * 100) if total > 0 else 0:.1f}%")
    print()

    # Show output directory structure
    print("OUTPUT STRUCTURE:")
    print(f"  data_parsed/")

    parsed_dir = Path("data_parsed")
    if parsed_dir.exists():
        for ticker_dir in sorted(parsed_dir.iterdir()):
            if ticker_dir.is_dir():
                print(f"    {ticker_dir.name}/")

                edgar_dir = ticker_dir / "edgar"
                if edgar_dir.exists():
                    jsonl_files = list(edgar_dir.glob("*.jsonl"))
                    tables_dir = edgar_dir / "tables"
                    table_count = len(list(tables_dir.glob("*.csv"))) if tables_dir.exists() else 0
                    print(f"      edgar/ ({len(jsonl_files)} documents, {table_count} tables)")

                earnings_dir = ticker_dir / "earnings"
                if earnings_dir.exists():
                    jsonl_files = list(earnings_dir.glob("*.jsonl"))
                    print(f"      earnings/ ({len(jsonl_files)} transcripts)")

    print()
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Review parsed files in data_parsed/ directory")
    print("2. Check JSONL format:")
    print("   head -1 data_parsed/AAPL/edgar/*.jsonl | jq")
    print("3. Verify Markdown exports are readable:")
    print("   cat data_parsed/AAPL/earnings/*.md | head -50")
    print("4. Load JSONL files into RAG/vector database")
    print("5. Test semantic search with section/speaker filtering")
    print()


def main():
    """Run full migration"""
    start_time = datetime.now()

    print()
    print("=" * 70)
    print("FULL MIGRATION TO NORMALIZED JSONL FORMAT")
    print("=" * 70)
    print()
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Migrate EDGAR filings
    edgar_stats = migrate_edgar_filings()

    # Migrate earnings transcripts
    transcript_stats = migrate_earnings_transcripts()

    # Generate report
    generate_migration_report(edgar_stats, transcript_stats)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print()

    # Return success if no failures
    total_failed = edgar_stats[1] + transcript_stats[1]
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
