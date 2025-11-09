"""
PDF Transcript Parser

Parses earnings call transcripts from PDF files.
"""

import logging
from pathlib import Path
from typing import Optional, Dict
import re

try:
    from pypdf import PdfReader
except ImportError:
    # Fallback to older PyPDF2 if pypdf not available
    from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


class PDFTranscriptParser:
    """Parser for earnings call transcript PDFs"""

    def parse_pdf(self, pdf_path: str) -> Optional[Dict]:
        """
        Parse PDF file and extract transcript text

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with transcript text and metadata
        """
        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return None

        logger.info(f"Parsing PDF: {pdf_file.name}")

        try:
            # Parse filename to get metadata
            metadata = self._parse_filename(pdf_file.name)

            if not metadata:
                logger.error(f"Could not parse filename: {pdf_file.name}")
                return None

            # Extract text from PDF
            text = self._extract_text_from_pdf(pdf_path)

            if not text:
                logger.error(f"No text extracted from PDF: {pdf_file.name}")
                return None

            # Clean the text
            cleaned_text = self._clean_text(text)

            return {
                'ticker': metadata['ticker'],
                'year': metadata['year'],
                'quarter': metadata['quarter'],
                'transcript': cleaned_text,
                'source_file': pdf_file.name,
                'source_path': str(pdf_file.absolute()),
                'file_size': pdf_file.stat().st_size,
            }

        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """
        Parse filename to extract ticker, year, quarter

        Expected format: {TICKER}_FY{year}_Q{quarter}.pdf
        Example: AAPL_FY2024_Q3.pdf
        """
        # Remove .pdf extension
        name = filename.replace('.pdf', '')

        # Pattern: TICKER_FY####_Q#
        pattern = r'^([A-Z.]+)_FY(\d{4})_Q([1-4])$'
        match = re.match(pattern, name)

        if match:
            ticker = match.group(1)
            year = int(match.group(2))
            quarter = int(match.group(3))

            return {
                'ticker': ticker,
                'year': year,
                'quarter': quarter
            }

        logger.warning(f"Filename doesn't match expected pattern: {filename}")
        return None

    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract all text from PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text or None if failed
        """
        try:
            reader = PdfReader(pdf_path)

            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
                    continue

            if not text_parts:
                return None

            return '\n\n'.join(text_parts)

        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Remove page numbers (common pattern: just a number on a line)
        text = re.sub(r'\n\d+\n', '\n', text)

        # Normalize line breaks
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')

        # Remove leading/trailing whitespace
        text = text.strip()

        return text


# Global parser instance
pdf_parser = PDFTranscriptParser()
