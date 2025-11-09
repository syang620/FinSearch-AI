#!/usr/bin/env python3
"""
Test script for EDGAR data ingestion

This script ingests SEC EDGAR filings for a few companies to test the system.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.data_ingestion.edgar.edgar_ingestion import edgar_ingestion

def main():
    print("=" * 60)
    print("FinSearch AI - EDGAR Data Ingestion Test")
    print("=" * 60)
    print()

    # Test with a few major companies
    test_tickers = ['AAPL', 'MSFT']
    num_filings = 2

    print(f"Testing ingestion for: {', '.join(test_tickers)}")
    print(f"Number of filings per company: {num_filings}")
    print()
    print("This may take several minutes...")
    print("-" * 60)
    print()

    try:
        # Run the ingestion
        result = edgar_ingestion.batch_ingest_companies(
            tickers=test_tickers,
            num_filings=num_filings
        )

        # Print results
        print()
        print("=" * 60)
        print("INGESTION RESULTS")
        print("=" * 60)
        print()
        print(f"✓ Success: {result['success']}")
        print(f"✓ Companies processed: {result['companies_processed']}")
        print(f"✓ Total filings: {result['total_filings']}")
        print(f"✓ Total chunks created: {result['total_chunks']}")
        print()

        # Detailed results
        print("Detailed Results:")
        print("-" * 60)
        for company_result in result['results']:
            if company_result['success']:
                print(f"\n{company_result['ticker']} (CIK: {company_result['cik']})")
                print(f"  Filings processed: {company_result['filings_processed']}")
                print(f"  Chunks created: {company_result['chunks_created']}")
                for filing in company_result['filings']:
                    print(f"    - {filing['form_type']} ({filing['filing_date']}): {filing['chunks']} chunks")
            else:
                print(f"\n{company_result['ticker']}: FAILED")
                print(f"  Error: {company_result.get('error', 'Unknown error')}")

        print()
        print("=" * 60)
        print("Test ingestion complete!")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print(f"Failed to run ingestion: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
