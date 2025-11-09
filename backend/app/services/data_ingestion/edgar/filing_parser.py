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
        Convert HTML to structured text while preserving semantic structure

        Preserves:
        - Headings (converted to markdown-style)
        - Lists (converted to bullet/numbered format)
        - Tables (basic structure preserved)
        - Paragraph breaks

        Removes:
        - Boilerplate (SEC headers, page numbers, disclaimers)
        - Script and style tags
        - Excessive whitespace

        Args:
            html_content: Raw HTML content

        Returns:
            Structured text with preserved semantic elements
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script, style, and navigation elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Convert structural elements to text with markers
            self._convert_headings(soup)
            self._convert_lists(soup)
            self._convert_tables(soup)

            # Get text
            text = soup.get_text(separator='\n')

            # Remove SEC boilerplate
            text = self._remove_boilerplate(text)

            # Clean up whitespace while preserving structure
            text = self._clean_whitespace(text)

            return text

        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            return html_content

    def _convert_headings(self, soup: BeautifulSoup) -> None:
        """Convert HTML headings to markdown-style markers"""
        for level in range(1, 7):  # h1-h6
            for heading in soup.find_all(f'h{level}'):
                heading_text = heading.get_text().strip()
                if heading_text:
                    # Convert to markdown style
                    markdown_heading = '\n' + ('#' * level) + ' ' + heading_text + '\n'
                    heading.replace_with(markdown_heading)

    def _convert_lists(self, soup: BeautifulSoup) -> None:
        """Convert HTML lists to text lists"""
        # Ordered lists
        for ol in soup.find_all('ol'):
            items = ol.find_all('li', recursive=False)
            list_text = '\n'.join([f"{i+1}. {item.get_text().strip()}" for i, item in enumerate(items)])
            ol.replace_with('\n' + list_text + '\n')

        # Unordered lists
        for ul in soup.find_all('ul'):
            items = ul.find_all('li', recursive=False)
            list_text = '\n'.join([f"• {item.get_text().strip()}" for item in items])
            ul.replace_with('\n' + list_text + '\n')

    def _convert_tables(self, soup: BeautifulSoup) -> None:
        """Convert HTML tables to text format"""
        for table in soup.find_all('table'):
            # Extract table caption if present
            caption = table.find('caption')
            caption_text = f"\n[TABLE: {caption.get_text().strip()}]\n" if caption else "\n[TABLE]\n"

            # Extract rows
            rows = []
            for tr in table.find_all('tr'):
                cells = []
                for cell in tr.find_all(['th', 'td']):
                    cells.append(cell.get_text().strip())
                if cells:
                    rows.append(' | '.join(cells))

            table_text = caption_text + '\n'.join(rows) + '\n[/TABLE]\n'
            table.replace_with(table_text)

    def _remove_boilerplate(self, text: str) -> str:
        """Remove SEC filing boilerplate and standard legal text"""
        # SEC header patterns
        boilerplate_patterns = [
            # SEC header information
            r'UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION.*?(?=\n\n|\nITEM)',
            r'Washington,\s*D\.?C\.?\s*20549',
            r'FORM\s+10-[KQ]\s*\n',
            # Page numbers and footers
            r'\n\s*\d+\s*\n',  # Standalone page numbers
            r'\n\s*Page\s+\d+\s+of\s+\d+\s*\n',
            # Standard disclaimers (beginning of line)
            r'^\s*Table of Contents\s*\n',
            # SEC filing codes
            r'Commission [Ff]ile [Nn]umber:.*?\n',
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '\n', text, flags=re.IGNORECASE | re.MULTILINE)

        return text

    def _clean_whitespace(self, text: str) -> str:
        """Clean whitespace while preserving paragraph structure"""
        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.splitlines()]

        # Remove leading/trailing whitespace
        text = '\n'.join(lines).strip()

        return text

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
