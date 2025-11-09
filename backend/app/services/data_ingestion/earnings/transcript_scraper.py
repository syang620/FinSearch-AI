"""
Seeking Alpha Transcript Scraper

Scrapes full earnings call transcript text from Seeking Alpha pages.
"""

import requests
import time
import logging
import re
from typing import Optional, Dict
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class TranscriptScraper:
    """Scraper for Seeking Alpha transcript pages"""

    RATE_LIMIT_DELAY = 5  # seconds between requests

    def __init__(self, user_agent: Optional[str] = None):
        """
        Initialize transcript scraper

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
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Upgrade-Insecure-Requests': '1',
        }

    def _rate_limit(self):
        """Apply rate limiting between requests"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def scrape_transcript(self, url: str) -> Optional[Dict]:
        """
        Scrape full transcript from Seeking Alpha page

        Args:
            url: Seeking Alpha transcript URL

        Returns:
            Dictionary with transcript text and metadata, or None if failed
        """
        # Handle relative URLs
        if url.startswith('/'):
            url = f"https://seekingalpha.com{url}"

        logger.info(f"Scraping transcript from: {url}")

        try:
            # Apply rate limiting
            self._rate_limit()

            # Make request
            response = self.session.get(url, headers=self._get_headers(), timeout=30)

            if response.status_code == 403:
                logger.warning(f"Access forbidden (403) for URL: {url}")
                return None

            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract transcript data
            result = self._extract_transcript_data(soup, url)

            if result:
                logger.info(
                    f"Successfully scraped transcript "
                    f"({len(result.get('transcript', ''))} characters)"
                )
            else:
                logger.warning(f"Failed to extract transcript from: {url}")

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping transcript from {url}: {e}")
            return None

    def _extract_transcript_data(
        self,
        soup: BeautifulSoup,
        url: str
    ) -> Optional[Dict]:
        """
        Extract transcript data from parsed HTML

        Args:
            soup: BeautifulSoup object
            url: Source URL

        Returns:
            Dictionary with transcript text and metadata
        """
        try:
            # Extract title
            title = self._extract_title(soup)

            # Extract date
            publish_date = self._extract_date(soup)

            # Extract participants
            participants = self._extract_participants(soup)

            # Extract transcript text
            transcript_text = self._extract_transcript_text(soup)

            if not transcript_text:
                return None

            # Parse quarter and year from title
            quarter_year = self._parse_quarter_year(title)

            return {
                'title': title,
                'publish_date': publish_date,
                'transcript': transcript_text,
                'participants': participants,
                'url': url,
                'year': quarter_year.get('year'),
                'quarter': quarter_year.get('quarter'),
            }

        except Exception as e:
            logger.error(f"Error extracting transcript data: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title"""
        title_tag = soup.find('h1')
        if title_tag:
            return title_tag.get_text(strip=True)

        # Try meta title
        meta_title = soup.find('meta', property='og:title')
        if meta_title:
            return meta_title.get('content', '')

        return ''

    def _extract_date(self, soup: BeautifulSoup) -> str:
        """Extract publish date"""
        # Try time tag
        time_tag = soup.find('time')
        if time_tag:
            return time_tag.get('datetime', time_tag.get_text(strip=True))

        # Try meta date
        meta_date = soup.find('meta', property='article:published_time')
        if meta_date:
            return meta_date.get('content', '')

        return ''

    def _extract_participants(self, soup: BeautifulSoup) -> list:
        """Extract list of participants/speakers"""
        participants = []

        # Look for participant sections (varies by Seeking Alpha layout)
        # This is a simple approach - may need refinement
        speaker_tags = soup.find_all(['strong', 'b'], string=re.compile(r'^[A-Z][a-z]+ [A-Z][a-z]+.*:'))

        for tag in speaker_tags:
            speaker = tag.get_text(strip=True).rstrip(':')
            if speaker and speaker not in participants:
                participants.append(speaker)

        return participants[:10]  # Limit to first 10 unique participants

    def _extract_transcript_text(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract the main transcript text

        This looks for common Seeking Alpha transcript patterns
        """
        # Try to find the article body
        # Seeking Alpha often uses specific classes/IDs - these may change

        # Method 1: Look for article body
        article_body = soup.find('div', {'data-test-id': 'article-content'})
        if not article_body:
            article_body = soup.find('article')
        if not article_body:
            article_body = soup.find('div', class_=re.compile(r'article|content|body'))

        if article_body:
            # Extract all text from paragraphs
            paragraphs = article_body.find_all(['p', 'div'], recursive=True)
            text_parts = []

            for p in paragraphs:
                text = p.get_text(separator=' ', strip=True)
                if text and len(text) > 20:  # Filter out very short lines
                    text_parts.append(text)

            if text_parts:
                return '\n\n'.join(text_parts)

        # Method 2: Fallback - get all paragraph text
        paragraphs = soup.find_all('p')
        text_parts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]

        if text_parts:
            return '\n\n'.join(text_parts)

        return None

    def _parse_quarter_year(self, title: str) -> Dict[str, Optional[int]]:
        """
        Parse quarter and year from title

        Examples:
            "Apple Q4 2024 Earnings Call" -> {quarter: 4, year: 2024}
            "AAPL Q1 FY2025 Results" -> {quarter: 1, year: 2025}
        """
        quarter = None
        year = None

        # Look for quarter pattern (Q1, Q2, Q3, Q4)
        quarter_match = re.search(r'Q([1-4])', title, re.IGNORECASE)
        if quarter_match:
            quarter = int(quarter_match.group(1))

        # Look for year pattern (2024, 2025, FY2024, etc.)
        year_match = re.search(r'(?:FY)?(\d{4})', title)
        if year_match:
            year = int(year_match.group(1))

        return {'quarter': quarter, 'year': year}


# Global scraper instance
transcript_scraper = TranscriptScraper()
