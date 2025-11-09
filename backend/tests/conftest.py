"""
Pytest fixtures and configuration for tests
"""

import pytest
from pathlib import Path
import json
import tempfile
import shutil


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for testing"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_10k_html():
    """Sample 10-K HTML content for testing"""
    return """
    <?xml version='1.0' encoding='ASCII'?>
    <html xmlns="http://www.w3.org/1999/xhtml">
    <head><title>10-K Filing</title></head>
    <body>
        <div>
            <p>ITEM 1. BUSINESS</p>
            <p>This company operates in the technology sector...</p>
        </div>
        <div>
            <p>ITEM 1A. RISK FACTORS</p>
            <p>The company faces various risks including...</p>
        </div>
        <div>
            <p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</p>
            <p>Revenue increased by 15% year over year...</p>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_10q_html():
    """Sample 10-Q HTML content for testing"""
    return """
    <html>
    <body>
        <div>
            <p>PART I. FINANCIAL INFORMATION</p>
            <p>Quarterly financial data...</p>
        </div>
        <div>
            <p>ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS</p>
            <p>This quarter showed strong performance...</p>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_filing_metadata():
    """Sample filing metadata"""
    return {
        "accession_number": "0000320193-24-000079",
        "filing_date": "2024-11-01",
        "report_date": "2024-09-30",
        "form_type": "10-K",
        "primary_document": "aapl-20240930.htm",
    }


@pytest.fixture
def test_companies():
    """Sample company list for testing"""
    return [
        {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Information Technology",
            "cik": "0000320193",
        },
        {
            "ticker": "MSFT",
            "company_name": "Microsoft Corporation",
            "sector": "Information Technology",
            "cik": "0000789019",
        },
    ]


@pytest.fixture
def mock_cik_mapping():
    """Mock CIK mapping data"""
    return {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "GOOGL": "0001652044",
        "AMZN": "0001018724",
    }


@pytest.fixture
def mock_submissions_response():
    """Mock SEC submissions API response"""
    return {
        "cik": "320193",
        "name": "Apple Inc.",
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-24-000079",
                    "0000320193-24-000050",
                    "0000320193-24-000030",
                ],
                "filingDate": [
                    "2024-11-01",
                    "2024-08-01",
                    "2024-05-02",
                ],
                "reportDate": [
                    "2024-09-30",
                    "2024-06-30",
                    "2024-03-31",
                ],
                "form": [
                    "10-K",
                    "10-Q",
                    "10-Q",
                ],
                "primaryDocument": [
                    "aapl-20240930.htm",
                    "aapl-20240630.htm",
                    "aapl-20240331.htm",
                ],
                "primaryDocDescription": [
                    "10-K",
                    "10-Q",
                    "10-Q",
                ],
                "fileNumber": [
                    "001-36743",
                    "001-36743",
                    "001-36743",
                ],
            }
        },
    }


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Cleanup any test data after tests"""
    yield
    # Cleanup happens automatically with tmp_path
