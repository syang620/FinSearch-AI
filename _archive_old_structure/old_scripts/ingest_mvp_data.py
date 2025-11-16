#!/usr/bin/env python3
"""
MVP Data Ingestion Script

Ingests SEC EDGAR filings for top 10 S&P 500 companies.
Saves data to filesystem for reprocessability.

Scope:
- Top 10 companies by market cap
- Past 8 quarters (2 years) of data
- 10-K/10-Q filings from SEC EDGAR
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.data_ingestion.top_companies import get_top_tickers, QUARTERS_TO_FETCH
from app.services.data_ingestion.edgar.edgar_client import edgar_client
from app.services.data_ingestion.edgar.edgar_ingestion import edgar_ingestion
from app.services.data_ingestion.file_storage import file_storage
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_edgar_data(ticker: str, num_filings: int = 8) -> dict:
    """
    Ingest EDGAR filings for a company

    Args:
        ticker: Stock ticker
        num_filings: Number of recent filings to fetch

    Returns:
        Ingestion results
    """
    logger.info(f"=== Ingesting EDGAR data for {ticker} ===")

    try:
        # Get CIK
        cik = edgar_client.get_cik_by_ticker(ticker)
        if not cik:
            logger.error(f"Could not find CIK for {ticker}")
            return {'success': False, 'error': 'CIK not found'}

        logger.info(f"Found CIK: {cik}")

        # Get filings
        filings = edgar_client.get_10k_10q_filings(cik, limit=num_filings)

        if not filings:
            logger.warning(f"No filings found for {ticker}")
            return {'success': True, 'filings_processed': 0}

        logger.info(f"Found {len(filings)} filings")

        # Process each filing - SAVE TO DISK
        processed = 0
        for filing in filings:
            result = edgar_ingestion._process_filing_with_storage(
                ticker=ticker,
                cik=cik,
                filing=filing,
                save_to_disk=True,   # Save files
                ingest_to_rag=False  # Don't ingest to RAG yet (for MVP)
            )

            if result['success']:
                processed += 1
                logger.info(
                    f"  ✓ {filing['form_type']} ({filing['filing_date']}) - "
                    f"saved to disk"
                )
            else:
                logger.error(f"  ✗ Failed: {result.get('error')}")

        return {
            'success': True,
            'filings_processed': processed,
            'total_available': len(filings)
        }

    except Exception as e:
        logger.error(f"Error ingesting EDGAR data for {ticker}: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Main ingestion flow"""
    print("=" * 70)
    print("FinSearch AI - MVP Data Ingestion")
    print("=" * 70)
    print()
    print("Scope:")
    print(f"  • Top 10 S&P 500 companies by market cap")
    print(f"  • {QUARTERS_TO_FETCH} quarters of data (~2 years)")
    print(f"  • EDGAR filings (10-K/10-Q)")
    print()
    print("Storage: Filesystem (data/edgar/)")
    print()
    print("-" * 70)
    print()

    # Get top companies
    tickers = get_top_tickers()
    logger.info(f"Top 10 companies: {', '.join(tickers)}")

    # Summary statistics
    total_edgar_filings = 0
    successful_companies = 0
    failed_companies = []

    # Process each company
    for i, ticker in enumerate(tickers, 1):
        print()
        print(f"[{i}/{len(tickers)}] Processing {ticker}...")
        print("-" * 70)

        try:
            # Ingest EDGAR data
            edgar_result = ingest_edgar_data(ticker, num_filings=QUARTERS_TO_FETCH)

            if edgar_result['success']:
                total_edgar_filings += edgar_result.get('filings_processed', 0)
                successful_companies += 1
            else:
                failed_companies.append(ticker)

        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
            failed_companies.append(ticker)

    # Final summary
    print()
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print()
    print(f"Companies processed: {successful_companies}/{len(tickers)}")
    print(f"EDGAR filings saved: {total_edgar_filings}")
    print()

    if failed_companies:
        print(f"Failed companies: {', '.join(failed_companies)}")
        print()

    print("Data saved to:")
    print(f"  • EDGAR filings: {file_storage.edgar_dir}")
    print()
    print("Next steps:")
    print("  1. Inspect saved files in data/ directory")
    print("  2. Run RAG ingestion to load into vector database")
    print("  3. Test queries via chat interface")
    print()
    print("=" * 70)

    return 0 if not failed_companies else 1


if __name__ == "__main__":
    sys.exit(main())
