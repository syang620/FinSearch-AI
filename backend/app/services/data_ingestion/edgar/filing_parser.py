"""
SEC Filing Parser

Parses HTML and text from SEC filing documents (10-K, 10-Q) and extracts clean text.
"""

from bs4 import BeautifulSoup
import re
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FilingParser:
    """Parse SEC filing documents"""

    def __init__(self):
        pass

    def clean_html(self, html_content: str) -> str:
        """
        Remove HTML tags and clean up text

        Args:
            html_content: Raw HTML content

        Returns:
            Clean text
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text

        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            return html_content

    def extract_sections(self, text: str, filing_type: str) -> Dict[str, str]:
        """
        Extract major sections from 10-K or 10-Q filing

        Args:
            text: Filing text
            filing_type: '10-K' or '10-Q'

        Returns:
            Dictionary of section name to content
        """
        sections = {}

        try:
            if filing_type == '10-K':
                # 10-K has Parts I-IV with items
                section_patterns = {
                    'business': r'ITEM\s+1\.?\s+BUSINESS',
                    'risk_factors': r'ITEM\s+1A\.?\s+RISK\s+FACTORS',
                    'properties': r'ITEM\s+2\.?\s+PROPERTIES',
                    'legal_proceedings': r'ITEM\s+3\.?\s+LEGAL\s+PROCEEDINGS',
                    'financial_data': r'ITEM\s+6\.?\s+SELECTED\s+FINANCIAL\s+DATA',
                    'md_and_a': r'ITEM\s+7\.?\s+MANAGEMENT.?S\s+DISCUSSION\s+AND\s+ANALYSIS',
                    'financial_statements': r'ITEM\s+8\.?\s+FINANCIAL\s+STATEMENTS',
                }
            else:  # 10-Q
                section_patterns = {
                    'financial_information': r'PART\s+I\.?\s+FINANCIAL\s+INFORMATION',
                    'financial_statements': r'ITEM\s+1\.?\s+FINANCIAL\s+STATEMENTS',
                    'md_and_a': r'ITEM\s+2\.?\s+MANAGEMENT.?S\s+DISCUSSION\s+AND\s+ANALYSIS',
                    'quantitative_disclosures': r'ITEM\s+3\.?\s+QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES',
                    'controls': r'ITEM\s+4\.?\s+CONTROLS\s+AND\s+PROCEDURES',
                }

            # Find sections
            for section_name, pattern in section_patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = match.start()
                    # Find next section or end of document
                    # Look for next ITEM or PART
                    next_section = re.search(
                        r'(ITEM|PART)\s+\d+[A-Z]?\.?',
                        text[start + len(match.group()):],
                        re.IGNORECASE
                    )

                    if next_section:
                        end = start + len(match.group()) + next_section.start()
                    else:
                        end = len(text)

                    section_text = text[start:end].strip()
                    sections[section_name] = section_text

                    logger.debug(f"Extracted section '{section_name}' ({len(section_text)} chars)")

        except Exception as e:
            logger.error(f"Error extracting sections: {e}")

        return sections

    def parse_filing(
        self,
        html_content: str,
        filing_type: str = '10-K'
    ) -> Dict[str, any]:
        """
        Parse a filing document

        Args:
            html_content: Raw HTML content of filing
            filing_type: Type of filing ('10-K' or '10-Q')

        Returns:
            Dictionary with parsed data
        """
        try:
            # Clean HTML
            text = self.clean_html(html_content)

            # Extract sections
            sections = self.extract_sections(text, filing_type)

            # Get metadata
            metadata = self._extract_metadata(text)

            result = {
                'full_text': text,
                'sections': sections,
                'metadata': metadata,
                'filing_type': filing_type,
                'text_length': len(text),
                'num_sections': len(sections),
            }

            logger.info(
                f"Parsed {filing_type} filing: "
                f"{len(text)} chars, {len(sections)} sections"
            )

            return result

        except Exception as e:
            logger.error(f"Error parsing filing: {e}")
            return {
                'full_text': html_content,
                'sections': {},
                'metadata': {},
                'filing_type': filing_type,
                'error': str(e)
            }

    def _extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract metadata from filing text"""
        metadata = {}

        try:
            # Extract company name
            company_match = re.search(
                r'COMPANY\s+CONFORMED\s+NAME:\s+(.+)',
                text,
                re.IGNORECASE
            )
            if company_match:
                metadata['company_name'] = company_match.group(1).strip()

            # Extract CIK
            cik_match = re.search(
                r'CENTRAL\s+INDEX\s+KEY:\s+(\d+)',
                text,
                re.IGNORECASE
            )
            if cik_match:
                metadata['cik'] = cik_match.group(1).strip()

            # Extract filing date
            date_match = re.search(
                r'FILED\s+AS\s+OF\s+DATE:\s+(\d+)',
                text,
                re.IGNORECASE
            )
            if date_match:
                metadata['filing_date'] = date_match.group(1).strip()

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")

        return metadata

    def get_key_sections_text(
        self,
        sections: Dict[str, str],
        section_names: Optional[list] = None
    ) -> str:
        """
        Get concatenated text from key sections

        Args:
            sections: Dictionary of sections
            section_names: List of section names to include (None = all)

        Returns:
            Concatenated text from specified sections
        """
        if section_names is None:
            section_names = sections.keys()

        texts = []
        for section_name in section_names:
            if section_name in sections:
                texts.append(f"### {section_name.upper()} ###\n\n{sections[section_name]}")

        return "\n\n".join(texts)


# Singleton instance
filing_parser = FilingParser()
