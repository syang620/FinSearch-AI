"""
Simple File-Based Storage for MVP

Saves raw and parsed documents to filesystem with simple JSON metadata.
No database required for MVP.
"""

from pathlib import Path
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FileStorage:
    """Simple file-based storage manager"""

    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.edgar_dir = self.base_dir / "edgar"
        self.earnings_dir = self.base_dir / "earnings_calls"
        self._ensure_directories()

    def _ensure_directories(self):
        """Create base directories if they don't exist"""
        self.edgar_dir.mkdir(parents=True, exist_ok=True)
        self.earnings_dir.mkdir(parents=True, exist_ok=True)

    def _get_company_dir(self, source: str, ticker: str) -> Path:
        """Get directory for a company"""
        if source == "edgar":
            base = self.edgar_dir
        elif source == "earnings":
            base = self.earnings_dir
        else:
            raise ValueError(f"Unknown source: {source}")

        company_dir = base / ticker
        company_dir.mkdir(parents=True, exist_ok=True)
        return company_dir

    def save_edgar_filing(
        self,
        ticker: str,
        form_type: str,
        date: str,
        raw_content: str,
        parsed_text: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Save EDGAR filing (both raw and parsed)

        Args:
            ticker: Stock ticker
            form_type: '10-K' or '10-Q'
            date: Filing date (YYYY-MM-DD)
            raw_content: Raw HTML content
            parsed_text: Cleaned text
            metadata: Additional metadata

        Returns:
            Dict with file paths
        """
        try:
            company_dir = self._get_company_dir("edgar", ticker)

            # Determine filename
            year = date.split('-')[0]
            if form_type == '10-K':
                filename_base = f"10-K_{year}"
            else:  # 10-Q
                # Determine quarter from month
                month = int(date.split('-')[1])
                quarter = (month - 1) // 3 + 1
                filename_base = f"10-Q_{year}_Q{quarter}"

            # Save raw HTML
            raw_file = company_dir / f"raw_{filename_base}.htm"
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(raw_content)

            # Save parsed text
            parsed_file = company_dir / f"{filename_base}.txt"
            with open(parsed_file, 'w', encoding='utf-8') as f:
                f.write(parsed_text)

            # Update metadata
            self._update_edgar_metadata(ticker, {
                "type": form_type,
                "date": date,
                "year": year,
                "quarter": quarter if form_type == '10-Q' else None,
                "raw_file": str(raw_file.name),
                "parsed_file": str(parsed_file.name),
                "rag_ingested": False,
                "saved_at": datetime.now().isoformat(),
                **metadata
            })

            logger.info(f"Saved {form_type} for {ticker}: {parsed_file}")

            return {
                "raw_file": str(raw_file),
                "parsed_file": str(parsed_file)
            }

        except Exception as e:
            logger.error(f"Error saving EDGAR filing: {e}")
            raise

    def save_earnings_call(
        self,
        ticker: str,
        year: int,
        quarter: int,
        transcript: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Save earnings call transcript

        Args:
            ticker: Stock ticker
            year: Year
            quarter: Quarter (1-4)
            transcript: Transcript text
            metadata: Additional metadata

        Returns:
            Path to saved file
        """
        try:
            company_dir = self._get_company_dir("earnings", ticker)

            # Save transcript
            filename = f"{year}_Q{quarter}.txt"
            filepath = company_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(transcript)

            # Update metadata
            self._update_earnings_metadata(ticker, {
                "year": year,
                "quarter": quarter,
                "file": filename,
                "rag_ingested": False,
                "saved_at": datetime.now().isoformat(),
                **metadata
            })

            logger.info(f"Saved earnings call for {ticker} {year} Q{quarter}")

            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving earnings call: {e}")
            raise

    def _update_edgar_metadata(self, ticker: str, filing_data: Dict):
        """Update EDGAR metadata JSON file"""
        company_dir = self._get_company_dir("edgar", ticker)
        metadata_file = company_dir / "metadata.json"

        # Load existing metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "ticker": ticker,
                "filings": []
            }

        # Add or update filing
        filings = metadata.get("filings", [])

        # Remove existing entry for same date if exists
        filings = [f for f in filings if f.get("date") != filing_data.get("date")]
        filings.append(filing_data)

        # Sort by date (newest first)
        filings.sort(key=lambda x: x.get("date", ""), reverse=True)
        metadata["filings"] = filings

        # Save
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _update_earnings_metadata(self, ticker: str, call_data: Dict):
        """Update earnings call metadata JSON file"""
        company_dir = self._get_company_dir("earnings", ticker)
        metadata_file = company_dir / "metadata.json"

        # Load existing metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "ticker": ticker,
                "earnings_calls": []
            }

        # Add or update call
        calls = metadata.get("earnings_calls", [])

        # Remove existing entry for same quarter if exists
        quarter_key = f"{call_data['year']}_Q{call_data['quarter']}"
        calls = [c for c in calls if f"{c['year']}_Q{c['quarter']}" != quarter_key]
        calls.append(call_data)

        # Sort by date (newest first)
        calls.sort(key=lambda x: (x.get("year", 0), x.get("quarter", 0)), reverse=True)
        metadata["earnings_calls"] = calls

        # Save
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_parsed_files(self, ticker: str, source: str = "edgar") -> List[Dict[str, str]]:
        """
        Get all parsed files for a company

        Args:
            ticker: Stock ticker
            source: 'edgar' or 'earnings'

        Returns:
            List of dicts with file info
        """
        company_dir = self._get_company_dir(source, ticker)

        if source == "edgar":
            pattern = "*.txt"
            exclude = "raw_"
        else:
            pattern = "*.txt"
            exclude = None

        files = []
        for filepath in company_dir.glob(pattern):
            if exclude and filepath.name.startswith(exclude):
                continue

            files.append({
                "path": str(filepath),
                "name": filepath.name,
                "size": filepath.stat().st_size
            })

        return files

    def mark_as_ingested(self, ticker: str, source: str, filename: str):
        """Mark a file as RAG ingested"""
        company_dir = self._get_company_dir(source, ticker)
        metadata_file = company_dir / "metadata.json"

        if not metadata_file.exists():
            logger.warning(f"No metadata file for {ticker}")
            return

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Update the appropriate entry
        if source == "edgar":
            for filing in metadata.get("filings", []):
                if filing.get("parsed_file") == filename:
                    filing["rag_ingested"] = True
                    filing["rag_ingested_at"] = datetime.now().isoformat()
        else:  # earnings
            for call in metadata.get("earnings_calls", []):
                if call.get("file") == filename:
                    call["rag_ingested"] = True
                    call["rag_ingested_at"] = datetime.now().isoformat()

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


# Singleton instance
file_storage = FileStorage()
