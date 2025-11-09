"""
SEC EDGAR API Client

Implements SEC EDGAR REST API access with proper rate limiting and headers.
SEC requires user-agent header and enforces 10 requests/second limit.
"""

import requests
import time
from typing import Dict, List, Optional
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class EDGARClient:
    """Client for SEC EDGAR API"""

    BASE_URL = "https://data.sec.gov"
    SUBMISSIONS_URL = f"{BASE_URL}/submissions"
    RATE_LIMIT_DELAY = 0.1  # 100ms between requests (10 requests/second)

    def __init__(self, user_agent: str = "FinSearch AI research@example.com"):
        """
        Initialize EDGAR client

        Args:
            user_agent: User agent string (SEC requires format: Company Name contact@email.com)
        """
        self.user_agent = user_agent
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            
        }
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting (10 requests/second)"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.RATE_LIMIT_DELAY:
            sleep_time = self.RATE_LIMIT_DELAY - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with rate limiting"""
        try:
            self._rate_limit()
            logger.info(f"Requesting: {url}")

            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            return response

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    def get_ticker_to_cik_mapping(self) -> Dict[str, str]:
        """
        Get mapping of ticker symbols to CIK numbers

        Returns:
            Dictionary mapping tickers to CIK numbers
        """
        try:
            # Note: company_tickers.json is on www.sec.gov, not data.sec.gov
            url = "https://www.sec.gov/files/company_tickers.json"
            response = self._make_request(url)

            if response:
                data = response.json()
                # Convert to ticker: CIK mapping
                mapping = {}
                for entry in data.values():
                    ticker = entry['ticker']
                    # CIK needs to be 10 digits with leading zeros
                    cik = str(entry['cik_str']).zfill(10)
                    mapping[ticker] = cik

                logger.info(f"Fetched {len(mapping)} ticker-CIK mappings")
                return mapping

            return {}

        except Exception as e:
            logger.error(f"Error fetching ticker-CIK mapping: {e}")
            return {}

    def get_cik_by_ticker(self, ticker: str) -> Optional[str]:
        """
        Get CIK number for a ticker symbol

        Args:
            ticker: Stock ticker symbol

        Returns:
            10-digit CIK number or None
        """
        mapping = self.get_ticker_to_cik_mapping()
        return mapping.get(ticker.upper())

    def get_company_submissions(self, cik: str) -> Optional[Dict]:
        """
        Get all submissions for a company

        Args:
            cik: Company CIK number (can be with or without leading zeros)

        Returns:
            Company submissions data or None
        """
        try:
            # Ensure CIK is 10 digits with leading zeros
            cik_padded = str(cik).zfill(10)

            url = f"{self.SUBMISSIONS_URL}/CIK{cik_padded}.json"
            response = self._make_request(url)

            if response:
                data = response.json()
                logger.info(f"Fetched submissions for CIK {cik_padded}")
                return data

            return None

        except Exception as e:
            logger.error(f"Error fetching submissions for CIK {cik}: {e}")
            return None

    def get_company_filings(
        self,
        cik: str,
        form_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get filings for a company, optionally filtered by form type

        Args:
            cik: Company CIK number
            form_types: List of form types to filter (e.g., ['10-K', '10-Q'])

        Returns:
            List of filing dictionaries
        """
        submissions = self.get_company_submissions(cik)

        if not submissions or 'filings' not in submissions:
            return []

        try:
            recent_filings = submissions['filings']['recent']

            # Extract filing data
            filings = []
            num_filings = len(recent_filings.get('accessionNumber', []))

            for i in range(num_filings):
                filing = {
                    'accession_number': recent_filings['accessionNumber'][i],
                    'filing_date': recent_filings['filingDate'][i],
                    'report_date': recent_filings['reportDate'][i],
                    'form_type': recent_filings['form'][i],
                    'file_number': recent_filings['fileNumber'][i],
                    'primary_document': recent_filings['primaryDocument'][i],
                    'primary_doc_description': recent_filings['primaryDocDescription'][i],
                }
                filings.append(filing)

            # Filter by form types if specified
            if form_types:
                filings = [f for f in filings if f['form_type'] in form_types]

            logger.info(f"Found {len(filings)} filings for CIK {cik}")
            return filings

        except Exception as e:
            logger.error(f"Error processing filings: {e}")
            return []

    def get_filing_document_url(
        self,
        cik: str,
        accession_number: str,
        primary_document: str
    ) -> str:
        """
        Construct URL for filing document

        Args:
            cik: Company CIK number
            accession_number: Filing accession number
            primary_document: Primary document filename

        Returns:
            URL to the filing document
        """
        # Remove dashes from accession number for URL
        accession_no_dashes = accession_number.replace('-', '')
        cik_padded = str(cik).zfill(10)

        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_padded}/{accession_no_dashes}/{primary_document}"
        )

        return url

    def download_filing_text(
        self,
        cik: str,
        accession_number: str,
        primary_document: str
    ) -> Optional[str]:
        """
        Download filing document text

        Args:
            cik: Company CIK number
            accession_number: Filing accession number
            primary_document: Primary document filename

        Returns:
            Document text or None
        """
        try:
            url = self.get_filing_document_url(cik, accession_number, primary_document)
            response = self._make_request(url)

            if response:
                return response.text

            return None

        except Exception as e:
            logger.error(f"Error downloading filing: {e}")
            return None

    def get_10k_10q_filings(
        self,
        cik: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get 10-K and 10-Q filings for a company

        Args:
            cik: Company CIK number
            limit: Maximum number of filings to return

        Returns:
            List of 10-K and 10-Q filings
        """
        filings = self.get_company_filings(cik, form_types=['10-K', '10-Q'])

        if limit:
            filings = filings[:limit]

        return filings


# Singleton instance - will be initialized with config email
def get_edgar_client():
    """Get EDGAR client with proper user agent from config"""
    from app.core.config import settings
    user_agent = f"FinSearch AI {settings.EDGAR_USER_EMAIL or 'research@example.com'}"
    return EDGARClient(user_agent)

edgar_client = get_edgar_client()
