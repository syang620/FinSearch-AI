"""
Tests for EDGAR client
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.data_ingestion.edgar.edgar_client import EDGARClient
import time


class TestEDGARClient:
    """Test EDGARClient class"""

    def test_initialization(self):
        """Test client initialization"""
        client = EDGARClient(user_agent="Test Agent test@example.com")

        assert client.user_agent == "Test Agent test@example.com"
        assert "User-Agent" in client.headers
        assert client.headers["User-Agent"] == "Test Agent test@example.com"

    def test_rate_limiting(self):
        """Test that rate limiting enforces delays"""
        client = EDGARClient()

        # First request should not delay
        start = time.time()
        client._rate_limit()
        first_duration = time.time() - start

        assert first_duration < 0.05  # Should be very fast

        # Second request should delay
        start = time.time()
        client._rate_limit()
        second_duration = time.time() - start

        # Should wait at least 0.1 seconds (rate limit delay)
        assert second_duration >= 0.09  # Allow small margin

    @patch('app.services.data_ingestion.edgar.edgar_client.requests.get')
    def test_get_ticker_to_cik_mapping(self, mock_get, mock_cik_mapping):
        """Test fetching ticker to CIK mapping"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "0": {"ticker": "AAPL", "cik_str": 320193},
            "1": {"ticker": "MSFT", "cik_str": 789019},
            "2": {"ticker": "GOOGL", "cik_str": 1652044},
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EDGARClient()
        mapping = client.get_ticker_to_cik_mapping()

        assert "AAPL" in mapping
        assert mapping["AAPL"] == "0000320193"
        assert mapping["MSFT"] == "0000789019"
        assert mapping["GOOGL"] == "0001652044"

    @patch('app.services.data_ingestion.edgar.edgar_client.requests.get')
    def test_get_cik_by_ticker(self, mock_get):
        """Test getting CIK for a specific ticker"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "0": {"ticker": "AAPL", "cik_str": 320193},
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EDGARClient()
        cik = client.get_cik_by_ticker("AAPL")

        assert cik == "0000320193"

    @patch('app.services.data_ingestion.edgar.edgar_client.requests.get')
    def test_get_cik_by_ticker_not_found(self, mock_get):
        """Test getting CIK for non-existent ticker"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "0": {"ticker": "AAPL", "cik_str": 320193},
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EDGARClient()
        cik = client.get_cik_by_ticker("INVALID")

        assert cik is None

    @patch('app.services.data_ingestion.edgar.edgar_client.requests.get')
    def test_get_company_submissions(self, mock_get, mock_submissions_response):
        """Test fetching company submissions"""
        mock_response = Mock()
        mock_response.json.return_value = mock_submissions_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EDGARClient()
        submissions = client.get_company_submissions("0000320193")

        assert submissions is not None
        assert "filings" in submissions
        assert "recent" in submissions["filings"]

        # Check URL was correct
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "CIK0000320193.json" in call_args[0][0]

    @patch('app.services.data_ingestion.edgar.edgar_client.requests.get')
    def test_get_company_filings(self, mock_get, mock_submissions_response):
        """Test getting company filings"""
        mock_response = Mock()
        mock_response.json.return_value = mock_submissions_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EDGARClient()
        filings = client.get_company_filings("0000320193")

        assert len(filings) == 3
        assert all("accession_number" in f for f in filings)
        assert all("form_type" in f for f in filings)
        assert all("filing_date" in f for f in filings)

    @patch('app.services.data_ingestion.edgar.edgar_client.requests.get')
    def test_get_company_filings_filtered(self, mock_get, mock_submissions_response):
        """Test filtering filings by form type"""
        mock_response = Mock()
        mock_response.json.return_value = mock_submissions_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EDGARClient()
        filings = client.get_company_filings("0000320193", form_types=["10-K"])

        # Should only return 10-K (1 out of 3)
        assert len(filings) == 1
        assert filings[0]["form_type"] == "10-K"

    @patch('app.services.data_ingestion.edgar.edgar_client.requests.get')
    def test_get_10k_10q_filings(self, mock_get, mock_submissions_response):
        """Test getting only 10-K and 10-Q filings"""
        mock_response = Mock()
        mock_response.json.return_value = mock_submissions_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EDGARClient()
        filings = client.get_10k_10q_filings("0000320193")

        # All 3 in mock data are 10-K or 10-Q
        assert len(filings) == 3
        assert all(f["form_type"] in ["10-K", "10-Q"] for f in filings)

    @patch('app.services.data_ingestion.edgar.edgar_client.requests.get')
    def test_get_10k_10q_filings_with_limit(self, mock_get, mock_submissions_response):
        """Test limiting number of filings returned"""
        mock_response = Mock()
        mock_response.json.return_value = mock_submissions_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EDGARClient()
        filings = client.get_10k_10q_filings("0000320193", limit=2)

        assert len(filings) == 2

    def test_get_filing_document_url(self):
        """Test URL construction for filing document"""
        client = EDGARClient()

        url = client.get_filing_document_url(
            cik="0000320193",
            accession_number="0000320193-24-000079",
            primary_document="aapl-20240930.htm"
        )

        assert "https://www.sec.gov/Archives/edgar/data/" in url
        assert "0000320193" in url
        assert "000032019324000079" in url  # No dashes
        assert "aapl-20240930.htm" in url

    @patch('app.services.data_ingestion.edgar.edgar_client.requests.get')
    def test_download_filing_text(self, mock_get, sample_10k_html):
        """Test downloading filing document"""
        mock_response = Mock()
        mock_response.text = sample_10k_html
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EDGARClient()
        content = client.download_filing_text(
            cik="0000320193",
            accession_number="0000320193-24-000079",
            primary_document="aapl-20240930.htm"
        )

        assert content == sample_10k_html

    @patch('app.services.data_ingestion.edgar.edgar_client.requests.get')
    def test_download_filing_text_failure(self, mock_get):
        """Test handling download failure"""
        mock_get.side_effect = Exception("Network error")

        client = EDGARClient()
        content = client.download_filing_text(
            cik="0000320193",
            accession_number="0000320193-24-000079",
            primary_document="test.htm"
        )

        assert content is None

    @patch('app.services.data_ingestion.edgar.edgar_client.requests.get')
    def test_request_headers(self, mock_get):
        """Test that requests include proper headers"""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EDGARClient(user_agent="Test test@example.com")
        client.get_ticker_to_cik_mapping()

        # Check headers were passed
        call_args = mock_get.call_args
        headers = call_args[1]["headers"]

        assert "User-Agent" in headers
        assert headers["User-Agent"] == "Test test@example.com"
        assert "Accept-Encoding" in headers
