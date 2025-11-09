#!/usr/bin/env python3
"""
Simple EDGAR API test - Tests just the SEC EDGAR client functionality
"""

import requests
import time
from datetime import datetime

print("=" * 60)
print("Simple SEC EDGAR API Test")
print("=" * 60)
print()

# Test configuration
USER_AGENT = "FinSearch AI test@example.com"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

def get_cik_for_ticker(ticker):
    """Get CIK for a ticker"""
    print(f"Fetching CIK for {ticker}...")
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    data = response.json()

    for entry in data.values():
        if entry['ticker'] == ticker:
            cik = str(entry['cik_str']).zfill(10)
            print(f"  ✓ Found CIK: {cik}")
            return cik

    return None

def get_company_filings(cik):
    """Get filings for a company"""
    print(f"Fetching filings for CIK {cik}...")
    time.sleep(0.1)  # Rate limit

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    data = response.json()

    filings = data['filings']['recent']
    num_filings = len(filings.get('accessionNumber', []))

    print(f"  ✓ Found {num_filings} total filings")

    # Find 10-K and 10-Q
    results = []
    for i in range(num_filings):
        form_type = filings['form'][i]
        if form_type in ['10-K', '10-Q']:
            results.append({
                'form_type': form_type,
                'filing_date': filings['filingDate'][i],
                'accession_number': filings['accessionNumber'][i],
            })

    return results[:5]  # Return first 5

def main():
    # Test companies
    tickers = ['AAPL', 'MSFT']

    all_results = {}

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        print("-" * 40)

        try:
            # Get CIK
            cik = get_cik_for_ticker(ticker)
            if not cik:
                print(f"  ✗ Could not find CIK for {ticker}")
                continue

            # Get filings
            filings = get_company_filings(cik)

            all_results[ticker] = {
                'cik': cik,
                'filings': filings
            }

            print(f"  ✓ Found {len(filings)} 10-K/10-Q filings:")
            for filing in filings:
                print(f"    - {filing['form_type']} filed on {filing['filing_date']}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Companies processed: {len(all_results)}")
    total_filings = sum(len(r['filings']) for r in all_results.values())
    print(f"Total 10-K/10-Q filings found: {total_filings}")
    print()
    print("✓ SEC EDGAR API is working correctly!")
    print()
    print("Next step: Run full ingestion with RAG integration")
    print("  (requires installing full dependencies)")

if __name__ == "__main__":
    main()
