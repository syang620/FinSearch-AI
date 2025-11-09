"""
Seeking Alpha API Client

Fetches earnings call transcript links from Seeking Alpha API.
"""

import requests
import time
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SeekingAlphaClient:
    """Client for Seeking Alpha API v3"""

    BASE_URL = "https://seekingalpha.com/api/v3"
    RATE_LIMIT_DELAY = 5  # seconds between requests

    def __init__(self, user_agent: Optional[str] = None):
        """
        Initialize Seeking Alpha client

        Args:
            user_agent: Custom user agent string (optional)
        """
        self.user_agent = user_agent or (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        self.session = requests.Session()
        self._last_request_time = 0

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            'User-Agent': self.user_agent,
            'Accept': 'application/json',
            'Upgrade-Insecure-Requests': '1',
        }

    def _rate_limit(self):
        """Apply rate limiting between requests"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def get_transcripts_list(
        self,
        ticker: str,
        page_number: int = 1,
        page_size: int = 50
    ) -> Optional[Dict]:
        """
        Get list of earnings call transcripts for a ticker

        Args:
            ticker: Stock ticker symbol
            page_number: Page number for pagination
            page_size: Number of results per page

        Returns:
            JSON response with transcript metadata or None if failed
        """
        url = (
            f"{self.BASE_URL}/symbols/{ticker}/transcripts"
            f"?id={ticker}&page[size]={page_size}&page[number]={page_number}"
        )

        logger.info(f"Fetching transcripts for {ticker} (page {page_number})")

        try:
            # Apply rate limiting
            self._rate_limit()

            # Make request
            response = self.session.get(url, headers=self._get_headers(), timeout=30)

            # Check response
            if response.status_code == 403:
                logger.warning(
                    f"Access forbidden (403) - may need authentication. "
                    f"Ticker: {ticker}"
                )
                return None

            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching transcripts for {ticker}: {e}")
            return None

    def get_all_transcripts(
        self,
        ticker: str,
        max_pages: Optional[int] = None
    ) -> List[Dict]:
        """
        Get all transcripts for a ticker (all pages)

        Args:
            ticker: Stock ticker symbol
            max_pages: Maximum number of pages to fetch (None for all)

        Returns:
            List of transcript metadata dictionaries
        """
        transcripts = []
        page = 1

        while True:
            # Check max pages limit
            if max_pages and page > max_pages:
                break

            # Fetch page
            response = self.get_transcripts_list(ticker, page_number=page)

            if not response:
                break

            # Extract data
            data = response.get('data', [])
            if not data:
                break

            # Parse transcripts
            for item in data:
                transcript_info = self._parse_transcript_item(item)
                if transcript_info:
                    transcripts.append(transcript_info)

            # Check if more pages
            meta = response.get('meta', {})
            page_info = meta.get('page', {})
            total_pages = page_info.get('totalPages', 1)

            logger.info(f"Fetched page {page}/{total_pages} for {ticker}")

            if page >= total_pages:
                break

            page += 1

        logger.info(f"Total transcripts found for {ticker}: {len(transcripts)}")
        return transcripts

    def _parse_transcript_item(self, item: Dict) -> Optional[Dict]:
        """
        Parse a transcript item from API response

        Args:
            item: Transcript data item from API

        Returns:
            Parsed transcript metadata
        """
        try:
            attributes = item.get('attributes', {})
            links = item.get('links', {})

            return {
                'id': item.get('id'),
                'title': attributes.get('title', ''),
                'publish_date': attributes.get('publishOn', ''),
                'transcript_url': links.get('self', ''),
                'raw_data': item  # Keep raw data for reference
            }
        except Exception as e:
            logger.error(f"Error parsing transcript item: {e}")
            return None


# Global client instance
seekingalpha_client = SeekingAlphaClient()
