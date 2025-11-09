"""
EDGAR Filing Parser V2

Parses SEC EDGAR filings (10-K, 10-Q) from inline XBRL HTML to normalized JSONL format.

Key improvements over V1:
- Handles inline XBRL format correctly
- Preserves document hierarchy and section structure
- Extracts financial tables to CSV
- Generates both JSONL and Markdown outputs
- Paragraph-level units with complete metadata
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

from bs4 import BeautifulSoup
import html2text
import pandas as pd

# Import metadata schema helpers
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metadata_schema import compute_period, compute_chunk_id, get_current_timestamp

logger = logging.getLogger(__name__)


class EDGARParserV2:
    """Parser for SEC EDGAR filings with inline XBRL support"""

    # Section patterns for 10-K filings
    SECTIONS_10K = {
        'Item 1': r'(?i)ITEM\s*1[A-Z]?\b\.?\s*[-–—]?\s*BUSINESS',
        'Item 1A': r'(?i)ITEM\s*1A\b\.?\s*[-–—]?\s*RISK\s+FACTORS',
        'Item 1B': r'(?i)ITEM\s*1B\b\.?\s*[-–—]?\s*UNRESOLVED\s+STAFF',
        'Item 2': r'(?i)ITEM\s*2\b\.?\s*[-–—]?\s*PROPERTIES',
        'Item 3': r'(?i)ITEM\s*3\b\.?\s*[-–—]?\s*LEGAL\s+PROCEEDINGS',
        'Item 4': r'(?i)ITEM\s*4\b\.?\s*[-–—]?\s*MINE\s+SAFETY',
        'Item 5': r'(?i)ITEM\s*5\b\.?\s*[-–—]?\s*MARKET\s+FOR',
        'Item 6': r'(?i)ITEM\s*6\b\.?\s*[-–—]?\s*(?:SELECTED|RESERVED)',
        'Item 7': r"(?i)ITEM\s*7[A-Z]?\b\.?\s*[-–—]?\s*MANAGEMENT['\u2019]?S?\s+DISCUSSION",
        'Item 7A': r'(?i)ITEM\s*7A\b\.?\s*[-–—]?\s*QUANTITATIVE\s+AND\s+QUALITATIVE',
        'Item 8': r'(?i)ITEM\s*8\b\.?\s*[-–—]?\s*FINANCIAL\s+STATEMENTS',
        'Item 9': r'(?i)ITEM\s*9[ABC]?\b\.?\s*[-–—]?\s*(?:CHANGES\s+IN|CONTROLS|DISCLOSURE)',
        'Item 10': r'(?i)ITEM\s*10\b\.?\s*[-–—]?\s*DIRECTORS',
        'Item 11': r'(?i)ITEM\s*11\b\.?\s*[-–—]?\s*EXECUTIVE\s+COMPENSATION',
        'Item 12': r'(?i)ITEM\s*12\b\.?\s*[-–—]?\s*SECURITY\s+OWNERSHIP',
        'Item 13': r'(?i)ITEM\s*13\b\.?\s*[-–—]?\s*CERTAIN\s+RELATIONSHIPS',
        'Item 14': r'(?i)ITEM\s*14\b\.?\s*[-–—]?\s*PRINCIPAL\s+(?:ACCOUNTANT|ACCOUNTING)',
        'Item 15': r'(?i)ITEM\s*15\b\.?\s*[-–—]?\s*EXHIBITS',
        'Item 16': r'(?i)ITEM\s*16\b\.?\s*[-–—]?\s*FORM\s+10-K',
    }

    # Section patterns for 10-Q filings
    SECTIONS_10Q = {
        'Part I': r'(?i)PART\s+I\.?\s*[-–—]?\s*FINANCIAL\s+INFORMATION',
        'Item 1': r'(?i)ITEM\s*1\b\.?\s*[-–—]?\s*FINANCIAL\s+STATEMENTS',
        'Item 2': r"(?i)ITEM\s*2\b\.?\s*[-–—]?\s*MANAGEMENT['\u2019]?S?\s+DISCUSSION",
        'Item 3': r'(?i)ITEM\s*3\b\.?\s*[-–—]?\s*QUANTITATIVE\s+AND\s+QUALITATIVE',
        'Item 4': r'(?i)ITEM\s*4\b\.?\s*[-–—]?\s*CONTROLS\s+AND\s+PROCEDURES',
        'Part II': r'(?i)PART\s+II\.?\s*[-–—]?\s*OTHER\s+INFORMATION',
        'Item 1A': r'(?i)ITEM\s*1A\b\.?\s*[-–—]?\s*RISK\s+FACTORS',
        'Item 5': r'(?i)ITEM\s*5\b\.?\s*[-–—]?\s*OTHER\s+INFORMATION',
        'Item 6': r'(?i)ITEM\s*6\b\.?\s*[-–—]?\s*EXHIBITS',
    }

    def __init__(self):
        """Initialize parser"""
        self.html_converter = html2text.HTML2Text()
        self.html_converter.body_width = 0  # No line wrapping
        self.html_converter.ignore_images = True
        self.html_converter.ignore_links = False
        self.html_converter.ignore_emphasis = False

    def parse_filing_to_jsonl(
        self,
        html_path: str,
        ticker: str,
        doc_type: str,
        filing_date: str,
        quarter: Optional[str] = None
    ) -> Tuple[List[Dict], List[Dict], str]:
        """
        Parse EDGAR filing to JSONL format

        Args:
            html_path: Path to raw HTML file
            ticker: Stock ticker symbol
            doc_type: Filing type ('10-K' or '10-Q')
            filing_date: Filing date (YYYY-MM-DD)
            quarter: Quarter (Q1-Q4 for 10-Q, None for 10-K)

        Returns:
            Tuple of (paragraph_units, table_metadata, markdown_export)
        """
        logger.info(f"Parsing {doc_type} filing: {html_path}")

        # Read HTML file
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()

        # Step 1: Clean XBRL HTML
        soup = self._clean_xbrl_html(html_content)

        # Extract fiscal year from filing_date
        fiscal_year = int(filing_date[:4])

        # Determine quarter for doc_id
        if doc_type == '10-K':
            quarter_str = 'FY'
            quarter_code = None  # For 10-K, quarter is None
        else:
            quarter_str = quarter if quarter else 'Q1'
            quarter_code = quarter_str

        # Step 2: Extract tables to CSV
        doc_id = f"{ticker}_{doc_type.replace('-', '')}_{fiscal_year}"
        if doc_type == '10-Q' and quarter_code:
            doc_id += f"_{quarter_code}"

        tables_metadata = self._extract_tables(soup, doc_id, ticker, Path(html_path).parent)

        # Step 3: Detect sections
        sections = self._detect_sections(soup, doc_type)

        # Step 4: Convert to markdown
        markdown_text = self._convert_to_markdown(soup)

        # Step 5: Extract paragraph units
        paragraph_units = self._extract_paragraphs(
            markdown_text,
            sections,
            ticker,
            doc_type,
            filing_date,
            doc_id,
            html_path,
            fiscal_year,
            quarter_code
        )

        # Step 6: Generate markdown export
        markdown_export = self._generate_markdown_export(
            paragraph_units,
            tables_metadata,
            ticker,
            doc_type,
            filing_date
        )

        logger.info(
            f"Parsed {doc_type}: {len(paragraph_units)} paragraphs, "
            f"{len(sections)} sections, {len(tables_metadata)} tables"
        )

        return paragraph_units, tables_metadata, markdown_export

    def _clean_xbrl_html(self, html_content: str) -> BeautifulSoup:
        """
        Clean inline XBRL HTML and remove metadata

        Args:
            html_content: Raw HTML content

        Returns:
            Cleaned BeautifulSoup object
        """
        # Use lxml parser for better XBRL handling
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove XBRL metadata sections
        for tag_name in ['ix:header', 'ix:hidden', 'script', 'style']:
            for element in soup.find_all(tag_name):
                element.decompose()

        # Remove navigation and footer elements
        for element in soup.find_all(['nav', 'header', 'footer']):
            element.decompose()

        # Remove elements by class/id (common in SEC filings)
        for class_name in ['header', 'footer', 'navigation', 'menu', 'nav']:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                element.decompose()

        return soup

    def _detect_sections(
        self,
        soup: BeautifulSoup,
        doc_type: str
    ) -> List[Tuple[str, str, int]]:
        """
        Detect section boundaries in the document

        Args:
            soup: BeautifulSoup object
            doc_type: Filing type ('10-K' or '10-Q')

        Returns:
            List of (section_id, section_title, position) tuples
        """
        sections = []
        text = soup.get_text()

        # Select section patterns based on document type
        patterns = self.SECTIONS_10K if doc_type == '10-K' else self.SECTIONS_10Q

        for section_id, pattern in patterns.items():
            matches = list(re.finditer(pattern, text))
            if matches:
                # Use first match
                match = matches[0]
                section_title = match.group(0).strip()
                position = match.start()
                sections.append((section_id, section_title, position))

        # Sort by position
        sections.sort(key=lambda x: x[2])

        logger.info(f"Detected {len(sections)} sections in {doc_type}")
        return sections

    def _extract_tables(
        self,
        soup: BeautifulSoup,
        doc_id: str,
        ticker: str,
        output_dir: Path
    ) -> List[Dict]:
        """
        Extract financial tables to CSV files

        Args:
            soup: BeautifulSoup object
            doc_id: Document ID
            ticker: Stock ticker
            output_dir: Directory to save tables

        Returns:
            List of table metadata dictionaries
        """
        tables_metadata = []

        try:
            # Create tables directory
            tables_dir = output_dir.parent.parent / "data_parsed" / ticker / "edgar" / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)

            # Extract all tables using pandas
            html_str = str(soup)
            tables = pd.read_html(html_str, displayed_only=False)

            logger.info(f"Found {len(tables)} tables in document")

            for idx, df in enumerate(tables):
                # Skip very small tables (likely not financial data)
                if df.shape[0] < 2 or df.shape[1] < 2:
                    continue

                # Generate table filename
                table_filename = f"{doc_id}_table_{idx + 1}.csv"
                table_path = tables_dir / table_filename

                # Save to CSV
                df.to_csv(table_path, index=False)

                # Create metadata
                table_meta = {
                    'table_id': f"{doc_id}_t{idx + 1}",
                    'filename': table_filename,
                    'path': str(table_path),
                    'rows': df.shape[0],
                    'columns': df.shape[1],
                    'size_bytes': table_path.stat().st_size
                }
                tables_metadata.append(table_meta)

                logger.debug(f"Extracted table {idx + 1}: {df.shape[0]}x{df.shape[1]}")

        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")

        return tables_metadata

    def _convert_to_markdown(self, soup: BeautifulSoup) -> str:
        """
        Convert HTML to markdown format

        Args:
            soup: BeautifulSoup object

        Returns:
            Markdown text
        """
        # Get HTML string
        html_str = str(soup)

        # Convert to markdown
        markdown = self.html_converter.handle(html_str)

        # Clean up excessive whitespace
        markdown = re.sub(r'\n\s*\n\s*\n+', '\n\n', markdown)

        return markdown.strip()

    def _extract_paragraphs(
        self,
        markdown_text: str,
        sections: List[Tuple[str, str, int]],
        ticker: str,
        doc_type: str,
        filing_date: str,
        doc_id: str,
        source_file: str,
        fiscal_year: int,
        quarter: Optional[str]
    ) -> List[Dict]:
        """
        Extract paragraph-level units from markdown text

        Args:
            markdown_text: Markdown formatted text
            sections: List of detected sections
            ticker: Stock ticker
            doc_type: Filing type
            filing_date: Filing date
            doc_id: Document ID
            source_file: Path to source HTML file
            fiscal_year: Fiscal year
            quarter: Quarter (Q1-Q4 or None for 10-K)

        Returns:
            List of paragraph unit dictionaries
        """
        paragraphs = []
        unit_index = 0

        # Compute standardized fields
        parsed_at = get_current_timestamp()
        quarter_str = quarter if quarter else 'FY'
        period = compute_period(fiscal_year, quarter_str)

        # Create section boundaries map
        section_map = {}
        for i, (section_id, section_title, position) in enumerate(sections):
            end_pos = sections[i + 1][2] if i + 1 < len(sections) else len(markdown_text)
            section_map[(position, end_pos)] = (section_id, section_title)

        # Split into paragraphs
        text_paragraphs = re.split(r'\n\s*\n+', markdown_text)

        current_pos = 0
        for para_text in text_paragraphs:
            para_text = para_text.strip()

            # Skip empty paragraphs
            if not para_text or len(para_text) < 20:
                current_pos += len(para_text) + 2
                continue

            # Find which section this paragraph belongs to
            section_id = None
            section_title = None
            for (start, end), (sid, stitle) in section_map.items():
                if start <= current_pos < end:
                    section_id = sid
                    section_title = stitle
                    break

            # Compute chunk_id
            chunk_id = compute_chunk_id(doc_id, unit_index, 'paragraph')

            # Create paragraph unit with unified metadata schema
            para_unit = {
                # Core identifiers
                'doc_id': doc_id,
                'chunk_id': chunk_id,

                # Company information
                'ticker': ticker,
                'company': ticker,  # Use ticker as company name

                # Document type and period
                'doc_type': doc_type,
                'fiscal_year': fiscal_year,
                'quarter': quarter_str,
                'period': period,
                'filing_date': filing_date,

                # Section/structure
                'section_id': section_id,
                'section_title': section_title,

                # Chunk information
                'unit_type': 'paragraph',
                'unit_index': unit_index,
                'text': para_text,
                'char_count': len(para_text),
                'word_count': len(para_text.split()),

                # Source tracking
                'source_file': source_file,
                'parsed_at': parsed_at,

                # Transcript-specific fields (null for EDGAR)
                'phase': None,
                'speaker_name': None,
                'speaker_role': None,
                'speaker_firm': None,
                'utterance_id': None,
                'utterance_type': None,
                'token_count': None,
                'exchange_id': None,
                'exchange_role': None
            }

            paragraphs.append(para_unit)
            unit_index += 1
            current_pos += len(para_text) + 2

        return paragraphs

    def _generate_markdown_export(
        self,
        paragraphs: List[Dict],
        tables: List[Dict],
        ticker: str,
        doc_type: str,
        filing_date: str
    ) -> str:
        """
        Generate human-readable markdown export

        Args:
            paragraphs: List of paragraph units
            tables: List of table metadata
            ticker: Stock ticker
            doc_type: Filing type
            filing_date: Filing date

        Returns:
            Markdown formatted document
        """
        lines = []

        # Header
        lines.append(f"# {ticker} {doc_type} - {filing_date}")
        lines.append("")
        lines.append(f"**Document ID**: {paragraphs[0]['doc_id'] if paragraphs else 'N/A'}")
        lines.append(f"**Ticker**: {ticker}")
        lines.append(f"**Filing Type**: {doc_type}")
        lines.append(f"**Filing Date**: {filing_date}")
        lines.append(f"**Paragraphs**: {len(paragraphs)}")
        lines.append(f"**Tables**: {len(tables)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Group paragraphs by section
        sections = {}
        for para in paragraphs:
            section_id = para.get('section_id', 'Unknown')
            section_title = para.get('section_title', 'Unknown Section')
            if section_id not in sections:
                sections[section_id] = {
                    'title': section_title,
                    'paragraphs': []
                }
            sections[section_id]['paragraphs'].append(para)

        # Write sections
        for section_id, section_data in sections.items():
            lines.append(f"## {section_id}: {section_data['title']}")
            lines.append("")

            for para in section_data['paragraphs']:
                lines.append(para['text'])
                lines.append("")

        # Append table references
        if tables:
            lines.append("---")
            lines.append("")
            lines.append("## Financial Tables")
            lines.append("")
            for table in tables:
                lines.append(f"- **{table['table_id']}**: {table['filename']} ({table['rows']}x{table['columns']})")
            lines.append("")

        return '\n'.join(lines)


# Global parser instance
edgar_parser_v2 = EDGARParserV2()
