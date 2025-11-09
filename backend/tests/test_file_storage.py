"""
Tests for file storage functionality
"""

import pytest
import json
from pathlib import Path
from app.services.data_ingestion.file_storage import FileStorage


class TestFileStorage:
    """Test FileStorage class"""

    def test_initialization(self, temp_data_dir):
        """Test FileStorage initialization creates directories"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        assert storage.edgar_dir.exists()
        assert storage.earnings_dir.exists()
        assert storage.edgar_dir == temp_data_dir / "edgar"
        assert storage.earnings_dir == temp_data_dir / "earnings_calls"

    def test_save_edgar_10k_filing(self, temp_data_dir, sample_10k_html):
        """Test saving a 10-K filing"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        result = storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content=sample_10k_html,
            parsed_text="Cleaned text from 10-K",
            metadata={
                "cik": "0000320193",
                "accession_number": "0000320193-24-000079",
                "report_date": "2024-09-30"
            }
        )

        # Check return value
        assert "raw_file" in result
        assert "parsed_file" in result

        # Check files exist
        company_dir = temp_data_dir / "edgar" / "AAPL"
        assert (company_dir / "raw_10-K_2024.htm").exists()
        assert (company_dir / "10-K_2024.txt").exists()
        assert (company_dir / "metadata.json").exists()

        # Check file contents
        with open(company_dir / "raw_10-K_2024.htm") as f:
            assert f.read() == sample_10k_html

        with open(company_dir / "10-K_2024.txt") as f:
            assert f.read() == "Cleaned text from 10-K"

    def test_save_edgar_10q_filing_with_quarter(self, temp_data_dir, sample_10q_html):
        """Test saving a 10-Q filing with proper quarter calculation"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        # Q2 filing (May = month 5, quarter 2)
        storage.save_edgar_filing(
            ticker="MSFT",
            form_type="10-Q",
            date="2024-05-02",
            raw_content=sample_10q_html,
            parsed_text="Q2 report",
            metadata={"cik": "0000789019"}
        )

        company_dir = temp_data_dir / "edgar" / "MSFT"
        assert (company_dir / "raw_10-Q_2024_Q2.htm").exists()
        assert (company_dir / "10-Q_2024_Q2.txt").exists()

    def test_edgar_metadata_creation(self, temp_data_dir, sample_10k_html):
        """Test that metadata.json is created with correct structure"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content=sample_10k_html,
            parsed_text="Cleaned text",
            metadata={
                "cik": "0000320193",
                "accession_number": "0000320193-24-000079"
            }
        )

        metadata_file = temp_data_dir / "edgar" / "AAPL" / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["ticker"] == "AAPL"
        assert "filings" in metadata
        assert len(metadata["filings"]) == 1

        filing = metadata["filings"][0]
        assert filing["type"] == "10-K"
        assert filing["date"] == "2024-11-01"
        assert filing["year"] == "2024"
        assert filing["raw_file"] == "raw_10-K_2024.htm"
        assert filing["parsed_file"] == "10-K_2024.txt"
        assert filing["rag_ingested"] is False
        assert "saved_at" in filing

    def test_save_multiple_filings_same_company(self, temp_data_dir, sample_10k_html, sample_10q_html):
        """Test saving multiple filings for same company"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        # Save 10-K
        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content=sample_10k_html,
            parsed_text="10-K text",
            metadata={"cik": "0000320193"}
        )

        # Save 10-Q
        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-Q",
            date="2024-08-01",
            raw_content=sample_10q_html,
            parsed_text="10-Q text",
            metadata={"cik": "0000320193"}
        )

        # Check both files exist
        company_dir = temp_data_dir / "edgar" / "AAPL"
        assert (company_dir / "10-K_2024.txt").exists()
        assert (company_dir / "10-Q_2024_Q3.txt").exists()

        # Check metadata has both
        metadata_file = company_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert len(metadata["filings"]) == 2

    def test_save_earnings_call(self, temp_data_dir, sample_transcript):
        """Test saving earnings call transcript"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        result = storage.save_earnings_call(
            ticker="AAPL",
            year=2024,
            quarter=3,
            transcript=sample_transcript,
            metadata={
                "date": "2024-08-01",
                "participants": ["CEO", "CFO", "Analysts"]
            }
        )

        # Check file exists
        filepath = Path(result)
        assert filepath.exists()
        assert filepath.name == "2024_Q3.txt"

        # Check content
        with open(filepath) as f:
            assert f.read() == sample_transcript

        # Check metadata
        metadata_file = temp_data_dir / "earnings_calls" / "AAPL" / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["ticker"] == "AAPL"
        assert len(metadata["earnings_calls"]) == 1

        call = metadata["earnings_calls"][0]
        assert call["year"] == 2024
        assert call["quarter"] == 3
        assert call["file"] == "2024_Q3.txt"
        assert call["rag_ingested"] is False

    def test_get_parsed_files_edgar(self, temp_data_dir, sample_10k_html):
        """Test getting list of parsed files"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        # Save a filing
        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content=sample_10k_html,
            parsed_text="Text",
            metadata={}
        )

        # Get parsed files
        files = storage.get_parsed_files("AAPL", source="edgar")

        assert len(files) == 1
        assert files[0]["name"] == "10-K_2024.txt"
        assert "path" in files[0]
        assert "size" in files[0]

    def test_get_parsed_files_excludes_raw(self, temp_data_dir, sample_10k_html):
        """Test that get_parsed_files excludes raw files"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content=sample_10k_html,
            parsed_text="Text",
            metadata={}
        )

        files = storage.get_parsed_files("AAPL", source="edgar")

        # Should only return parsed file, not raw
        assert len(files) == 1
        assert not files[0]["name"].startswith("raw_")

    def test_mark_as_ingested_edgar(self, temp_data_dir, sample_10k_html):
        """Test marking a filing as RAG ingested"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content=sample_10k_html,
            parsed_text="Text",
            metadata={}
        )

        # Mark as ingested
        storage.mark_as_ingested("AAPL", "edgar", "10-K_2024.txt")

        # Check metadata updated
        metadata_file = temp_data_dir / "edgar" / "AAPL" / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        filing = metadata["filings"][0]
        assert filing["rag_ingested"] is True
        assert "rag_ingested_at" in filing

    def test_mark_as_ingested_earnings(self, temp_data_dir, sample_transcript):
        """Test marking earnings call as RAG ingested"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        storage.save_earnings_call(
            ticker="AAPL",
            year=2024,
            quarter=3,
            transcript=sample_transcript,
            metadata={}
        )

        storage.mark_as_ingested("AAPL", "earnings", "2024_Q3.txt")

        metadata_file = temp_data_dir / "earnings_calls" / "AAPL" / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        call = metadata["earnings_calls"][0]
        assert call["rag_ingested"] is True

    def test_update_existing_filing(self, temp_data_dir, sample_10k_html):
        """Test that saving same filing again updates it"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        # Save first time
        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content=sample_10k_html,
            parsed_text="Original text",
            metadata={}
        )

        # Save again with updated content
        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content=sample_10k_html,
            parsed_text="Updated text",
            metadata={}
        )

        # Check only one filing in metadata
        metadata_file = temp_data_dir / "edgar" / "AAPL" / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert len(metadata["filings"]) == 1

        # Check content updated
        with open(temp_data_dir / "edgar" / "AAPL" / "10-K_2024.txt") as f:
            assert f.read() == "Updated text"

    def test_filings_sorted_by_date(self, temp_data_dir, sample_10q_html):
        """Test that filings in metadata are sorted by date (newest first)"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        # Save filings out of order
        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-Q",
            date="2024-02-01",
            raw_content=sample_10q_html,
            parsed_text="Q1",
            metadata={}
        )

        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-Q",
            date="2024-08-01",
            raw_content=sample_10q_html,
            parsed_text="Q3",
            metadata={}
        )

        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-Q",
            date="2024-05-01",
            raw_content=sample_10q_html,
            parsed_text="Q2",
            metadata={}
        )

        # Check sorted
        metadata_file = temp_data_dir / "edgar" / "AAPL" / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        dates = [f["date"] for f in metadata["filings"]]
        assert dates == ["2024-08-01", "2024-05-01", "2024-02-01"]
