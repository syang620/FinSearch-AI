"""
Earnings Call Transcript Fetcher

Fetches earnings call transcripts from discountingcashflows.com
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from .dcf_scraper import dcf_scraper

logger = logging.getLogger(__name__)


class EarningsCallFetcher:
    """
    Fetches earnings call transcripts from discountingcashflows.com

    Uses authenticated scraping to fetch full transcript text.
    """

    def __init__(self):
        """Initialize earnings call fetcher"""
        self.scraper = dcf_scraper

    def fetch_transcript(
        self,
        ticker: str,
        year: int,
        quarter: int
    ) -> Optional[Dict]:
        """
        Fetch a specific earnings call transcript

        Args:
            ticker: Stock ticker symbol
            year: Year (e.g., 2024)
            quarter: Quarter (1-4)

        Returns:
            Dictionary with transcript data or None if not found
        """
        logger.info(f"Fetching {ticker} Q{quarter} {year} earnings call")

        # Scrape transcript directly from DCF
        transcript_data = self.scraper.scrape_transcript(ticker, year, quarter)

        return transcript_data

    def fetch_recent_transcripts(
        self,
        ticker: str,
        num_quarters: int = 8
    ) -> List[Dict]:
        """
        Fetch recent earnings call transcripts

        Args:
            ticker: Stock ticker symbol
            num_quarters: Number of recent quarters to fetch

        Returns:
            List of transcript dictionaries
        """
        logger.info(f"Fetching {num_quarters} recent transcripts for {ticker}")

        # Generate list of recent quarters
        # Start from current quarter and go backwards
        current_year = datetime.now().year
        current_month = datetime.now().month
        current_quarter = (current_month - 1) // 3 + 1

        quarters_to_fetch = []
        year = current_year
        quarter = current_quarter

        for _ in range(num_quarters):
            quarters_to_fetch.append((year, quarter))

            # Move to previous quarter
            quarter -= 1
            if quarter < 1:
                quarter = 4
                year -= 1

        # Fetch each transcript
        transcripts = []
        for i, (year, quarter) in enumerate(quarters_to_fetch, 1):
            logger.info(
                f"Fetching transcript {i}/{num_quarters}: "
                f"{ticker} Q{quarter} {year}"
            )

            transcript_data = self.scraper.scrape_transcript(ticker, year, quarter)

            if transcript_data:
                transcripts.append(transcript_data)
            else:
                logger.warning(f"No transcript found for {ticker} Q{quarter} {year}")

        logger.info(f"Successfully fetched {len(transcripts)} transcripts for {ticker}")
        return transcripts


# Global fetcher instance
earnings_fetcher = EarningsCallFetcher()
