"""
Integration tests for data ingestion flow
"""

import pytest
from unittest.mock import patch, Mock
from pathlib import Path
import json

from app.services.data_ingestion.edgar.edgar_ingestion import EDGARIngestion
from app.services.data_ingestion.file_storage import FileStorage


class TestIntegration:
    """Integration tests for full data flow"""

    @patch('app.services.data_ingestion.edgar.edgar_client.edgar_client.get_cik_by_ticker')
    @patch('app.services.data_ingestion.edgar.edgar_client.edgar_client.get_company_filings')
    @patch('app.services.data_ingestion.edgar.edgar_client.edgar_client.download_filing_text')
    def test_full_edgar_ingestion_one_company(
        self,
        mock_download,
        mock_get_filings,
        mock_get_cik,
        temp_data_dir,
        sample_10k_html,
        sample_filing_metadata
    ):
        """Test full EDGAR ingestion flow for one company"""
        # Setup mocks
        mock_get_cik.return_value = "0000320193"
        mock_get_filings.return_value = [sample_filing_metadata]
        mock_download.return_value = sample_10k_html

        # Create storage with temp dir
        storage = FileStorage(base_dir=str(temp_data_dir))

        # Create ingestion instance
        ingestion = EDGARIngestion()
        ingestion.rag_retriever.vector_store = Mock()  # Mock vector store

        # Run ingestion with file storage
        result = ingestion._process_filing_with_storage(
            ticker="AAPL",
            cik="0000320193",
            filing=sample_filing_metadata,
            save_to_disk=True,
            ingest_to_rag=False
        )

        # Verify result
        assert result["success"] is True
        assert result["saved_to_disk"] is True

        # Verify files created
        company_dir = temp_data_dir / "edgar" / "AAPL"
        assert (company_dir / "metadata.json").exists()
        assert (company_dir / "10-K_2024.txt").exists()
        assert (company_dir / "raw_10-K_2024.htm").exists()

        # Verify metadata content
        with open(company_dir / "metadata.json") as f:
            metadata = json.load(f)
            assert metadata["ticker"] == "AAPL"
            assert len(metadata["filings"]) == 1

    @patch('app.services.data_ingestion.edgar.edgar_client.edgar_client.get_cik_by_ticker')
    @patch('app.services.data_ingestion.edgar.edgar_client.edgar_client.get_company_filings')
    @patch('app.services.data_ingestion.edgar.edgar_client.edgar_client.download_filing_text')
    def test_multiple_filings_same_company(
        self,
        mock_download,
        mock_get_filings,
        mock_get_cik,
        temp_data_dir,
        sample_10k_html,
        sample_10q_html
    ):
        """Test ingesting multiple filings for same company"""
        mock_get_cik.return_value = "0000320193"

        filings = [
            {
                "accession_number": "0000320193-24-000079",
                "filing_date": "2024-11-01",
                "report_date": "2024-09-30",
                "form_type": "10-K",
                "primary_document": "aapl-20240930.htm",
            },
            {
                "accession_number": "0000320193-24-000050",
                "filing_date": "2024-08-01",
                "report_date": "2024-06-30",
                "form_type": "10-Q",
                "primary_document": "aapl-20240630.htm",
            }
        ]

        mock_get_filings.return_value = filings
        mock_download.side_effect = [sample_10k_html, sample_10q_html]

        storage = FileStorage(base_dir=str(temp_data_dir))
        ingestion = EDGARIngestion()
        ingestion.rag_retriever.vector_store = Mock()

        # Process both filings
        for filing in filings:
            ingestion._process_filing_with_storage(
                ticker="AAPL",
                cik="0000320193",
                filing=filing,
                save_to_disk=True,
                ingest_to_rag=False
            )

        # Verify both files exist
        company_dir = temp_data_dir / "edgar" / "AAPL"
        assert (company_dir / "10-K_2024.txt").exists()
        assert (company_dir / "10-Q_2024_Q3.txt").exists()

        # Verify metadata has both
        with open(company_dir / "metadata.json") as f:
            metadata = json.load(f)
            assert len(metadata["filings"]) == 2

    def test_file_structure_creation(self, temp_data_dir):
        """Test that directory structure is created correctly"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        # Save a filing
        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content="<html>test</html>",
            parsed_text="test",
            metadata={}
        )

        # Verify structure
        assert (temp_data_dir / "edgar").exists()
        assert (temp_data_dir / "edgar" / "AAPL").exists()

    def test_reprocessing_same_filing(self, temp_data_dir, sample_10k_html):
        """Test that reprocessing same filing updates it correctly"""
        storage = FileStorage(base_dir=str(temp_data_dir))

        # Save first time
        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content=sample_10k_html,
            parsed_text="Original text",
            metadata={"version": 1}
        )

        # Save again (reprocess)
        storage.save_edgar_filing(
            ticker="AAPL",
            form_type="10-K",
            date="2024-11-01",
            raw_content=sample_10k_html,
            parsed_text="Updated text",
            metadata={"version": 2}
        )

        # Verify only one filing in metadata
        with open(temp_data_dir / "edgar" / "AAPL" / "metadata.json") as f:
            metadata = json.load(f)
            assert len(metadata["filings"]) == 1
            assert metadata["filings"][0]["version"] == 2

        # Verify content is updated
        with open(temp_data_dir / "edgar" / "AAPL" / "10-K_2024.txt") as f:
            assert f.read() == "Updated text"
