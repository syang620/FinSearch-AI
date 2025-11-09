"""
Data validation tests
"""

import pytest
from pathlib import Path
import json

from app.services.data_ingestion.top_companies import get_top_companies, get_top_tickers
from app.services.data_ingestion.file_storage import FileStorage


class TestDataValidation:
    """Tests for data validation"""

    def test_top_companies_list_not_empty(self):
        """Test that top companies list is populated"""
        companies = get_top_companies()

        assert len(companies) == 10
        assert all("ticker" in c for c in companies)
        assert all("company_name" in c for c in companies)

    def test_top_companies_have_required_fields(self):
        """Test that all companies have required fields"""
        companies = get_top_companies()

        required_fields = ["ticker", "company_name", "sector", "market_cap_b"]

        for company in companies:
            for field in required_fields:
                assert field in company, f"Missing field {field} in {company}"

    def test_top_tickers_match_companies(self):
        """Test that tickers list matches companies list"""
        companies = get_top_companies()
        tickers = get_top_tickers()

        assert len(tickers) == len(companies)

        company_tickers = [c["ticker"] for c in companies]
        assert set(tickers) == set(company_tickers)

    def test_file_size_constraints_edgar(self, temp_data_dir):
        """Test that saved EDGAR files have reasonable sizes"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        large_html = "<html>" + ("x" * 1_500_000) + "</html>"  # 1.5MB
        parsed_text = "x" * 500_000  # 500KB

        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content=large_html,
            parsed_text=parsed_text,
            metadata={}
        )

        # Check file sizes
        raw_file = temp_data_dir / "edgar" / "AAPL" / "raw_10-K_2024.htm"
        parsed_file = temp_data_dir / "edgar" / "AAPL" / "10-K_2024.txt"

        assert raw_file.stat().st_size > 1_000_000  # > 1MB
        assert parsed_file.stat().st_size > 100_000  # > 100KB

    def test_file_size_constraints_earnings(self, temp_data_dir):
        """Test that earnings transcripts have reasonable sizes"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        transcript = "Earnings call transcript. " * 1000  # Reasonable size

        storage.save_earnings_call(
            ticker="AAPL",
            year=2024,
            quarter=3,
            transcript=transcript,
            metadata={}
        )

        # Check file size
        file_path = temp_data_dir / "earnings_calls" / "AAPL" / "2024_Q3.txt"
        assert file_path.stat().st_size > 1000  # At least 1KB

    def test_metadata_json_structure_edgar(self, temp_data_dir):
        """Test that EDGAR metadata.json has correct structure"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content="<html>test</html>",
            parsed_text="test",
            metadata={"cik": "0000320193"}
        )

        metadata_file = temp_data_dir / "edgar" / "AAPL" / "metadata.json"

        # Validate JSON is parseable
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Check required top-level fields
        assert "ticker" in metadata
        assert "filings" in metadata
        assert isinstance(metadata["filings"], list)

        # Check filing structure
        filing = metadata["filings"][0]
        required_fields = ["type", "date", "year", "raw_file", "parsed_file", "rag_ingested"]

        for field in required_fields:
            assert field in filing, f"Missing field {field}"

    def test_metadata_json_structure_earnings(self, temp_data_dir):
        """Test that earnings metadata.json has correct structure"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        storage.save_earnings_call(
            ticker="AAPL",
            year=2024,
            quarter=3,
            transcript="test",
            metadata={"date": "2024-08-01"}
        )

        metadata_file = temp_data_dir / "earnings_calls" / "AAPL" / "metadata.json"

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Check required fields
        assert "ticker" in metadata
        assert "earnings_calls" in metadata

        call = metadata["earnings_calls"][0]
        required_fields = ["year", "quarter", "file", "rag_ingested"]

        for field in required_fields:
            assert field in call

    def test_no_duplicate_filings_in_metadata(self, temp_data_dir):
        """Test that metadata doesn't contain duplicate filings"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        # Save same filing twice
        for i in range(2):
            storage.save_edgar_filing(
                ticker="AAPL",
                form_type="10-K",
                date="2024-11-01",
                raw_content="<html>test</html>",
                parsed_text=f"version {i}",
                metadata={}
            )

        metadata_file = temp_data_dir / "edgar" / "AAPL" / "metadata.json"

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Should only have one filing
        assert len(metadata["filings"]) == 1

    def test_ticker_case_consistency(self):
        """Test that tickers are uppercase"""
        tickers = get_top_tickers()

        for ticker in tickers:
            assert ticker == ticker.upper(), f"Ticker {ticker} should be uppercase"
