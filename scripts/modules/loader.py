"""
Data loading module for financial documents.
Consolidates loading from various sources (EDGAR, earnings calls, etc.).
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a financial document with metadata"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    doc_type: str  # '10K', '10Q', 'earnings_call', etc.
    company: str
    date: str
    source_path: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_jsonl(self) -> str:
        """Convert to JSONL format"""
        return json.dumps(self.to_dict())


class DataLoader:
    """
    Unified data loader for all financial documents.
    Handles raw data, normalized JSONL, and processed chunks.
    """

    def __init__(self, base_path: Union[str, Path] = "data"):
        """
        Initialize the data loader.

        Args:
            base_path: Base directory containing data subdirectories
        """
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.interim_path = self.base_path / "interim"
        self.processed_path = self.base_path / "processed"

        # Ensure paths exist
        for path in [self.raw_path, self.interim_path, self.processed_path]:
            if not path.exists():
                logger.warning(f"Data path does not exist: {path}")

    def load_company_metadata(self) -> Dict[str, Any]:
        """Load S&P 500 company metadata"""
        metadata_file = self.raw_path / "sp500_companies.json"

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning("Company metadata file not found")
            return {}

    def list_available_companies(self) -> List[str]:
        """List all companies with available data"""
        companies = set()

        # Check normalized data
        if self.interim_path.exists():
            normalized_path = self.interim_path / "normalized"
            if normalized_path.exists():
                companies.update([d.name for d in normalized_path.iterdir() if d.is_dir()])

        # Check raw data
        if self.raw_path.exists():
            for source in ["edgar", "earnings_calls"]:
                source_path = self.raw_path / source
                if source_path.exists():
                    companies.update([d.name for d in source_path.iterdir() if d.is_dir()])

        return sorted(list(companies))

    def load_company_documents(
        self,
        company: str,
        doc_types: Optional[List[str]] = None,
        use_normalized: bool = True
    ) -> List[Document]:
        """
        Load all documents for a company.

        Args:
            company: Company ticker symbol
            doc_types: Optional list of document types to load
            use_normalized: Whether to use normalized JSONL data

        Returns:
            List of Document objects
        """
        documents = []

        if use_normalized:
            # Load from normalized JSONL
            documents.extend(self._load_normalized_documents(company, doc_types))
        else:
            # Load from raw data
            documents.extend(self._load_raw_documents(company, doc_types))

        logger.info(f"Loaded {len(documents)} documents for {company}")
        return documents

    def _load_normalized_documents(
        self,
        company: str,
        doc_types: Optional[List[str]] = None
    ) -> List[Document]:
        """Load documents from normalized JSONL format"""
        documents = []
        company_path = self.interim_path / "normalized" / company

        if not company_path.exists():
            logger.warning(f"No normalized data found for {company}")
            return documents

        # Process each source directory (edgar, earnings)
        for source_dir in company_path.iterdir():
            if not source_dir.is_dir():
                continue

            # Process JSONL files
            for jsonl_file in source_dir.glob("*.jsonl"):
                # Skip if doc_type filter is applied
                if doc_types:
                    file_type = self._infer_doc_type(jsonl_file.name)
                    if file_type not in doc_types:
                        continue

                # Read JSONL file
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)

                            # Create Document object
                            doc = Document(
                                content=data.get('content', ''),
                                metadata=data.get('metadata', {}),
                                doc_id=data.get('doc_id', jsonl_file.stem),
                                doc_type=data.get('doc_type', self._infer_doc_type(jsonl_file.name)),
                                company=company,
                                date=data.get('date', data.get('metadata', {}).get('date', '')),
                                source_path=str(jsonl_file)
                            )
                            documents.append(doc)

        return documents

    def _load_raw_documents(
        self,
        company: str,
        doc_types: Optional[List[str]] = None
    ) -> List[Document]:
        """Load documents from raw data files"""
        documents = []

        # Load EDGAR documents
        edgar_path = self.raw_path / "edgar" / company
        if edgar_path.exists():
            for file_path in edgar_path.glob("*.txt"):
                if doc_types:
                    file_type = self._infer_doc_type(file_path.name)
                    if file_type not in doc_types:
                        continue

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                doc = Document(
                    content=content,
                    metadata={'source': 'edgar', 'filename': file_path.name},
                    doc_id=file_path.stem,
                    doc_type=self._infer_doc_type(file_path.name),
                    company=company,
                    date=self._extract_date_from_filename(file_path.name),
                    source_path=str(file_path)
                )
                documents.append(doc)

        # Load earnings call documents
        earnings_path = self.raw_path / "earnings_calls" / company
        if earnings_path.exists():
            for file_path in earnings_path.glob("*.txt"):
                if doc_types:
                    file_type = self._infer_doc_type(file_path.name)
                    if file_type not in doc_types:
                        continue

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                doc = Document(
                    content=content,
                    metadata={'source': 'earnings_calls', 'filename': file_path.name},
                    doc_id=file_path.stem,
                    doc_type='earnings_call',
                    company=company,
                    date=self._extract_date_from_filename(file_path.name),
                    source_path=str(file_path)
                )
                documents.append(doc)

        return documents

    def load_chunks(
        self,
        company: str,
        doc_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load pre-computed chunks for a company.

        Args:
            company: Company ticker symbol
            doc_types: Optional list of document types

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        chunks_path = self.processed_path / "chunks" / company

        if not chunks_path.exists():
            logger.warning(f"No chunks found for {company}")
            return chunks

        # Load chunk files
        for chunk_file in chunks_path.rglob("*_chunks.jsonl"):
            if doc_types:
                file_type = self._infer_doc_type(chunk_file.name)
                if file_type not in doc_types:
                    continue

            with open(chunk_file, 'r') as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))

        logger.info(f"Loaded {len(chunks)} chunks for {company}")
        return chunks

    def _infer_doc_type(self, filename: str) -> str:
        """Infer document type from filename"""
        filename_lower = filename.lower()

        if '10k' in filename_lower:
            return '10K'
        elif '10q' in filename_lower:
            return '10Q'
        elif '8k' in filename_lower:
            return '8K'
        elif 'earnings' in filename_lower or 'call' in filename_lower:
            return 'earnings_call'
        elif 'q1' in filename_lower or 'q2' in filename_lower or 'q3' in filename_lower or 'q4' in filename_lower:
            return 'earnings_call'
        else:
            return 'other'

    def _extract_date_from_filename(self, filename: str) -> str:
        """Extract date from filename"""
        # Common patterns: 2024_Q1, 2024-Q1, 20240331, etc.
        import re

        # Try year_quarter pattern
        match = re.search(r'(20\d{2})[_-]?Q(\d)', filename)
        if match:
            year, quarter = match.groups()
            return f"{year}-Q{quarter}"

        # Try full date pattern YYYYMMDD
        match = re.search(r'(20\d{2})(\d{2})(\d{2})', filename)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"

        # Try year only
        match = re.search(r'(20\d{2})', filename)
        if match:
            return match.group(1)

        return ""

    def save_document(self, document: Document, format: str = "jsonl"):
        """
        Save a document to the normalized format.

        Args:
            document: Document object to save
            format: Output format ('jsonl' or 'json')
        """
        # Determine output path
        output_dir = self.interim_path / "normalized" / document.company / document.doc_type.lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            output_file = output_dir / f"{document.doc_id}.jsonl"
            with open(output_file, 'a') as f:
                f.write(document.to_jsonl() + '\n')
        else:
            output_file = output_dir / f"{document.doc_id}.json"
            with open(output_file, 'w') as f:
                json.dump(document.to_dict(), f, indent=2)

        logger.info(f"Saved document to {output_file}")

    def get_stats(self, company: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about available data"""
        stats = {
            'companies': [],
            'total_documents': 0,
            'total_chunks': 0,
            'by_type': {}
        }

        companies = [company] if company else self.list_available_companies()

        for comp in companies:
            docs = self.load_company_documents(comp)
            chunks = self.load_chunks(comp)

            comp_stats = {
                'company': comp,
                'documents': len(docs),
                'chunks': len(chunks),
                'doc_types': {}
            }

            # Count by document type
            for doc in docs:
                doc_type = doc.doc_type
                if doc_type not in comp_stats['doc_types']:
                    comp_stats['doc_types'][doc_type] = 0
                comp_stats['doc_types'][doc_type] += 1

                if doc_type not in stats['by_type']:
                    stats['by_type'][doc_type] = 0
                stats['by_type'][doc_type] += 1

            stats['companies'].append(comp_stats)
            stats['total_documents'] += len(docs)
            stats['total_chunks'] += len(chunks)

        return stats