"""
Discounting Cash Flows (DCF) Transcript Scraper

Scrapes earnings call transcripts from discountingcashflows.com
"""

import logging
import re
from typing import Optional, Dict
from bs4 import BeautifulSoup

from .dcf_auth import dcf_authenticator

logger = logging.getLogger(__name__)


class DCFTranscriptScraper:
    """Scraper for discountingcashflows.com earnings transcripts"""

    BASE_URL = "https://discountingcashflows.com"

    def __init__(self):
        """Initialize DCF scraper"""
        self.authenticator = dcf_authenticator

    def scrape_transcript(
        self,
        ticker: str,
        year: int,
        quarter: int
    ) -> Optional[Dict]:
        """
        Scrape earnings call transcript

        Args:
            ticker: Stock ticker symbol
            year: Year (e.g., 2024)
            quarter: Quarter (1-4)

        Returns:
            Dictionary with transcript data or None if failed
        """
        url = f"{self.BASE_URL}/company/{ticker}/transcripts/{year}/{quarter}/"

        logger.info(f"Scraping transcript from: {url}")

        # Get authenticated session
        session = self.authenticator.get_authenticated_session()
        if not session:
            logger.error("Failed to authenticate")
            return None

        try:
            # Fetch the page
            response = session.get(url, timeout=30)

            # Check if redirected to login
            if '/accounts/login' in response.url:
                logger.error("Not authenticated - redirected to login")
                # Try logging in again
                if self.authenticator.login(force_new=True):
                    session = self.authenticator.get_authenticated_session()
                    response = session.get(url, timeout=30)
                else:
                    return None

            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract transcript data
            result = self._extract_transcript_data(soup, ticker, year, quarter, url)

            if result:
                logger.info(
                    f"Successfully scraped transcript "
                    f"({len(result.get('transcript', ''))} characters)"
                )
            else:
                logger.warning(f"Failed to extract transcript from: {url}")

            return result

        except Exception as e:
            logger.error(f"Error scraping transcript from {url}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_transcript_data(
        self,
        soup: BeautifulSoup,
        ticker: str,
        year: int,
        quarter: int,
        url: str
    ) -> Optional[Dict]:
        """
        Extract transcript data from parsed HTML

        Args:
            soup: BeautifulSoup object
            ticker: Stock ticker
            year: Year
            quarter: Quarter
            url: Source URL

        Returns:
            Dictionary with transcript text and metadata
        """
        try:
            # Extract title
            title = self._extract_title(soup, ticker, year, quarter)

            # Extract date
            publish_date = self._extract_date(soup)

            # Extract company name
            company_name = self._extract_company_name(soup)

            # Extract participants
            participants = self._extract_participants(soup)

            # Extract transcript text - this is the key part
            transcript_text = self._extract_transcript_text(soup)

            if not transcript_text:
                logger.error("No transcript text found on page")
                return None

            return {
                'ticker': ticker,
                'company_name': company_name,
                'title': title,
                'publish_date': publish_date,
                'transcript': transcript_text,
                'participants': participants,
                'url': url,
                'year': year,
                'quarter': quarter,
            }

        except Exception as e:
            logger.error(f"Error extracting transcript data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_title(self, soup: BeautifulSoup, ticker: str, year: int, quarter: int) -> str:
        """Extract or construct title"""
        # Try h1 tag
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)

        # Try meta title
        meta_title = soup.find('meta', property='og:title')
        if meta_title:
            return meta_title.get('content', '')

        # Construct from ticker/year/quarter
        return f"{ticker} Q{quarter} {year} Earnings Call Transcript"

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

    def _extract_company_name(self, soup: BeautifulSoup) -> str:
        """Extract company name"""
        # Look for company name in title or heading
        h1 = soup.find('h1')
        if h1:
            text = h1.get_text(strip=True)
            # Extract company name before " - Q"
            match = re.search(r'^(.+?)\s*-\s*Q\d', text)
            if match:
                return match.group(1)

        return ''

    def _extract_participants(self, soup: BeautifulSoup) -> list:
        """Extract list of participants/speakers"""
        participants = []

        # Look for participant sections
        # DCF might have specific structure - adjust as needed
        speaker_tags = soup.find_all(['strong', 'b'])

        for tag in speaker_tags:
            text = tag.get_text(strip=True)
            # Look for speaker patterns like "John Doe:" or "John Doe -"
            if ':' in text or ' - ' in text:
                speaker = text.rstrip(':').rstrip(' -').strip()
                if speaker and len(speaker) < 50 and speaker not in participants:
                    participants.append(speaker)

        return participants[:15]  # Limit to first 15 unique participants

    def _extract_transcript_text(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract the main transcript text from the page

        This is the critical function - needs to identify the correct
        container for transcript content on DCF.
        """
        # Strategy 1: Look for common transcript container IDs/classes
        transcript_containers = [
            soup.find('div', {'id': 'transcript'}),
            soup.find('div', {'id': 'transcript-content'}),
            soup.find('div', {'class': 'transcript'}),
            soup.find('div', {'class': 'transcript-content'}),
            soup.find('article', {'class': 'transcript'}),
            soup.find('div', {'id': 'content'}),
        ]

        for container in transcript_containers:
            if container:
                text = self._extract_text_from_container(container)
                if text and len(text) > 500:  # Ensure substantial content
                    return text

        # Strategy 2: Look for main content area
        main_content = soup.find('main')
        if main_content:
            text = self._extract_text_from_container(main_content)
            if text and len(text) > 500:
                return text

        # Strategy 3: Look for article tag
        article = soup.find('article')
        if article:
            text = self._extract_text_from_container(article)
            if text and len(text) > 500:
                return text

        # Strategy 4: Find all paragraphs and combine
        # (fallback - less reliable but catches most cases)
        paragraphs = soup.find_all('p')
        if paragraphs:
            text_parts = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                # Filter out navigation, footer, header text
                if (len(text) > 30 and
                    'cookie' not in text.lower() and
                    'privacy' not in text.lower() and
                    'navigation' not in text.lower()):
                    text_parts.append(text)

            if text_parts:
                combined = '\n\n'.join(text_parts)
                if len(combined) > 500:
                    return combined

        logger.warning("Could not find transcript text with any strategy")
        return None

    def _extract_text_from_container(self, container) -> str:
        """Extract clean text from a container element"""
        if not container:
            return ""

        # Get all text, preserving paragraph structure
        paragraphs = container.find_all(['p', 'div'])

        if not paragraphs:
            # Fallback: get all text
            return container.get_text(separator='\n\n', strip=True)

        text_parts = []
        for elem in paragraphs:
            text = elem.get_text(separator=' ', strip=True)
            if len(text) > 20:  # Filter very short fragments
                text_parts.append(text)

        return '\n\n'.join(text_parts)


# Global scraper instance
dcf_scraper = DCFTranscriptScraper()
