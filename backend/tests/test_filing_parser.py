"""
Tests for filing parser
"""

import pytest
from app.services.data_ingestion.edgar.filing_parser import FilingParser


class TestFilingParser:
    """Test FilingParser class"""

    def test_clean_html(self, sample_10k_html):
        """Test HTML cleaning"""
        parser = FilingParser()
        clean_text = parser.clean_html(sample_10k_html)

        # Should remove HTML tags
        assert "<html>" not in clean_text
        assert "<body>" not in clean_text
        assert "<div>" not in clean_text

        # Should keep text content
        assert "ITEM 1. BUSINESS" in clean_text
        assert "technology sector" in clean_text

    def test_clean_html_removes_scripts(self):
        """Test that script tags are removed"""
        html_with_script = """
        <html>
        <script>alert('test');</script>
        <body>Actual content</body>
        </html>
        """

        parser = FilingParser()
        clean_text = parser.clean_html(html_with_script)

        assert "alert" not in clean_text
        assert "Actual content" in clean_text

    def test_extract_sections_10k(self, sample_10k_html):
        """Test extracting sections from 10-K"""
        parser = FilingParser()
        text = parser.clean_html(sample_10k_html)
        sections = parser.extract_sections(text, filing_type="10-K")

        assert "business" in sections
        assert "risk_factors" in sections
        assert "md_and_a" in sections

    def test_parse_filing_10k(self, sample_10k_html):
        """Test parsing complete 10-K filing"""
        parser = FilingParser()
        result = parser.parse_filing(sample_10k_html, filing_type="10-K")

        assert "full_text" in result
        assert "sections" in result
        assert "metadata" in result
        assert result["filing_type"] == "10-K"
        assert result["text_length"] > 0

    def test_parse_filing_10q(self, sample_10q_html):
        """Test parsing 10-Q filing"""
        parser = FilingParser()
        result = parser.parse_filing(sample_10q_html, filing_type="10-Q")

        assert result["filing_type"] == "10-Q"
        assert "full_text" in result

    def test_get_key_sections_text(self, sample_10k_html):
        """Test getting concatenated key sections"""
        parser = FilingParser()
        parsed = parser.parse_filing(sample_10k_html, filing_type="10-K")

        key_text = parser.get_key_sections_text(
            parsed["sections"],
            section_names=["business", "risk_factors"]
        )

        assert "BUSINESS" in key_text
        assert "RISK FACTORS" in key_text

    def test_get_key_sections_text_with_none(self, sample_10k_html):
        """Test getting all sections when section_names is None"""
        parser = FilingParser()
        parsed = parser.parse_filing(sample_10k_html, filing_type="10-K")

        all_text = parser.get_key_sections_text(parsed["sections"])

        # Should include all found sections
        assert len(all_text) > 0

    def test_parse_filing_error_handling(self):
        """Test error handling for malformed HTML"""
        parser = FilingParser()
        bad_html = "Not valid HTML at all!"

        result = parser.parse_filing(bad_html, filing_type="10-K")

        # Should still return a result with full_text
        assert "full_text" in result
        assert "error" not in result  # Should not error, just return raw text
