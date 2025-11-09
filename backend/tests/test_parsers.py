#!/usr/bin/env python3
"""
Test Script for Parser V2 Modules

Tests EDGAR and Transcript parsers on sample files to validate:
- XBRL HTML parsing
- Section detection
- Table extraction
- Speaker detection and Q&A pairing
- JSONL/Markdown output generation
"""

import sys
import os
from pathlib import Path
import logging

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


def test_edgar_parser():
    """Test EDGAR parser on a sample 10-K file"""
    print("=" * 70)
    print("TESTING EDGAR PARSER V2")
    print("=" * 70)
    print()

    # Find a sample 10-K file
    edgar_dir = Path("data/edgar/AAPL")
    if not edgar_dir.exists():
        print("ERROR: EDGAR directory not found")
        return False

    # Find raw HTML file
    raw_files = list(edgar_dir.glob("raw_10-K_*.htm"))
    if not raw_files:
        print("ERROR: No raw 10-K files found")
        return False

    raw_file = raw_files[0]
    print(f"Testing with file: {raw_file.name}")
    print(f"File size: {raw_file.stat().st_size:,} bytes")
    print()

    try:
        # Parse the filing
        paragraphs, tables, markdown = edgar_parser_v2.parse_filing_to_jsonl(
            html_path=str(raw_file),
            ticker="AAPL",
            doc_type="10-K",
            filing_date="2025-10-31"
        )

        # Validate results
        print("RESULTS:")
        print(f"  Paragraphs extracted: {len(paragraphs)}")
        print(f"  Tables extracted: {len(tables)}")
        print(f"  Markdown length: {len(markdown):,} characters")
        print()

        # Show section breakdown
        if paragraphs:
            sections = {}
            for para in paragraphs:
                section = para.get('section_id', 'Unknown')
                sections[section] = sections.get(section, 0) + 1

            print("SECTIONS DETECTED:")
            for section, count in sorted(sections.items()):
                print(f"  {section}: {count} paragraphs")
            print()

        # Show sample paragraph
        if paragraphs:
            print("SAMPLE PARAGRAPH:")
            sample = paragraphs[0]
            print(f"  Section: {sample.get('section_id')}")
            print(f"  Text: {sample.get('text')[:200]}...")
            print()

        # Show table info
        if tables:
            print("TABLES:")
            for table in tables[:3]:
                print(f"  {table['table_id']}: {table['rows']}x{table['columns']}")
            if len(tables) > 3:
                print(f"  ... and {len(tables) - 3} more tables")
            print()

        # Save parsed files
        print("Saving parsed files...")
        result = file_storage.save_parsed_edgar(
            ticker="AAPL",
            doc_id="AAPL_10K_2025",
            paragraphs=paragraphs,
            tables=tables,
            markdown_text=markdown
        )

        print(f"  JSONL: {result['jsonl_file']}")
        print(f"  Markdown: {result['md_file']}")
        print()

        # Validation checks
        print("VALIDATION:")
        print(f"  ✓ Paragraphs > 100: {len(paragraphs) > 100}")
        print(f"  ✓ Sections detected: {len(sections) > 3}")
        print(f"  ✓ Tables found: {len(tables) > 0}")
        print(f"  ✓ Markdown generated: {len(markdown) > 10000}")
        print()

        return len(paragraphs) > 100 and len(sections) > 3

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transcript_parser():
    """Test Transcript parser on a sample earnings call"""
    print("=" * 70)
    print("TESTING TRANSCRIPT PARSER V2")
    print("=" * 70)
    print()

    # Find a sample transcript PDF
    earnings_manual_dir = Path("data/earnings_calls_manual/AAPL")
    if not earnings_manual_dir.exists():
        print("ERROR: Earnings calls manual directory not found")
        return False

    # Find PDF file
    pdf_files = list(earnings_manual_dir.glob("*.pdf"))
    if not pdf_files:
        print("ERROR: No PDF files found")
        return False

    pdf_file = pdf_files[0]
    print(f"Testing with file: {pdf_file.name}")
    print(f"File size: {pdf_file.stat().st_size:,} bytes")
    print()

    try:
        # Parse the transcript
        utterances, markdown = transcript_parser_v2.parse_transcript_to_jsonl(
            pdf_path=str(pdf_file),
            ticker="AAPL",
            year=2025,
            quarter=4
        )

        # Validate results
        print("RESULTS:")
        print(f"  Utterances extracted: {len(utterances)}")
        print(f"  Markdown length: {len(markdown):,} characters")
        print()

        # Show phase breakdown
        if utterances:
            phases = {}
            roles = {}
            for utt in utterances:
                phase = utt.get('phase', 'Unknown')
                role = utt.get('speaker_role', 'Unknown')
                phases[phase] = phases.get(phase, 0) + 1
                roles[role] = roles.get(role, 0) + 1

            print("PHASES:")
            for phase, count in sorted(phases.items()):
                print(f"  {phase}: {count} utterances")
            print()

            print("SPEAKER ROLES:")
            for role, count in sorted(roles.items()):
                print(f"  {role}: {count} utterances")
            print()

        # Show Q&A exchanges
        exchanges = [u for u in utterances if u.get('exchange_id')]
        if exchanges:
            exchange_ids = set(u.get('exchange_id') for u in exchanges)
            print(f"Q&A EXCHANGES: {len(exchange_ids)}")
            print()

        # Show sample utterance
        if utterances:
            print("SAMPLE UTTERANCE:")
            sample = utterances[0]
            print(f"  Speaker: {sample.get('speaker_name')} ({sample.get('speaker_role')})")
            print(f"  Phase: {sample.get('phase')}")
            print(f"  Text: {sample.get('text')[:200]}...")
            print()

        # Save parsed files
        print("Saving parsed files...")
        result = file_storage.save_parsed_transcript(
            ticker="AAPL",
            year=2025,
            quarter=4,
            utterances=utterances,
            markdown_text=markdown
        )

        print(f"  JSONL: {result['jsonl_file']}")
        print(f"  Markdown: {result['md_file']}")
        print()

        # Validation checks
        print("VALIDATION:")
        print(f"  ✓ Utterances > 10: {len(utterances) > 10}")
        print(f"  ✓ Speakers detected: {len(roles) > 3}")
        print(f"  ✓ Q&A detected: {len(exchanges) > 0}")
        print(f"  ✓ Markdown generated: {len(markdown) > 5000}")
        print()

        return len(utterances) > 10 and len(roles) > 3

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print()
    print("=" * 70)
    print("PARSER V2 VALIDATION TESTS")
    print("=" * 70)
    print()

    results = {}

    # Test EDGAR parser
    results['edgar'] = test_edgar_parser()

    # Test Transcript parser
    results['transcript'] = test_transcript_parser()

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()

    for parser, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {parser.upper()}: {status}")

    print()

    if all(results.values()):
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED - check output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
