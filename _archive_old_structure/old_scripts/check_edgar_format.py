#!/usr/bin/env python3
"""
Check the actual format of SEC EDGAR filings
"""

import requests
import time

USER_AGENT = "FinSearch AI test@example.com"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

# Get Apple's filings
cik = "0000320193"
url = f"https://data.sec.gov/submissions/CIK{cik}.json"

print("Fetching filing metadata...")
response = requests.get(url, headers=HEADERS)
data = response.json()

filings = data['filings']['recent']

# Find first 10-K
for i in range(len(filings['form'])):
    if filings['form'][i] == '10-K':
        accession = filings['accessionNumber'][i]
        primary_doc = filings['primaryDocument'][i]
        filing_date = filings['filingDate'][i]

        print(f"\nFirst 10-K Filing:")
        print(f"  Date: {filing_date}")
        print(f"  Accession: {accession}")
        print(f"  Primary Document: {primary_doc}")
        print(f"  File Extension: {primary_doc.split('.')[-1]}")

        # Construct URL
        accession_no_dash = accession.replace('-', '')
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dash}/{primary_doc}"

        print(f"\nDocument URL:")
        print(f"  {doc_url}")

        # Download and check first 500 chars
        print(f"\nDownloading to check format...")
        time.sleep(0.1)
        doc_response = requests.get(doc_url, headers=HEADERS)
        content = doc_response.text

        print(f"\nContent Length: {len(content):,} characters")
        print(f"\nFirst 500 characters:")
        print("-" * 60)
        print(content[:500])
        print("-" * 60)

        # Check if it's HTML, XML, or plain text
        if content.strip().startswith('<?xml'):
            print("\nFormat: XML")
        elif content.strip().startswith('<'):
            print("\nFormat: HTML")
        else:
            print("\nFormat: Plain Text")

        # Check for XBRL
        if 'xmlns' in content[:1000] and 'xbrl' in content[:1000].lower():
            print("Contains: XBRL tags")

        break

print("\n" + "=" * 60)
print("Summary: SEC 10-K/10-Q filings are typically in HTML format")
print("         Some newer filings use XBRL (XML-based)")
print("=" * 60)
