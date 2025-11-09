"""
EDGAR Ingestion Orchestrator

Coordinates the ingestion of SEC EDGAR filings into the RAG system.
"""

from typing import List, Dict, Optional
import logging
from datetime import datetime
from app.services.data_ingestion.edgar.edgar_client import edgar_client
from app.services.data_ingestion.edgar.filing_parser import filing_parser
from app.services.data_ingestion.edgar.sp500_companies import sp500_companies
from app.services.data_ingestion.file_storage import file_storage

logger = logging.getLogger(__name__)


class EDGARIngestion:
    """Orchestrate EDGAR data ingestion"""

    def __init__(self):
        self.edgar_client = edgar_client
        self.filing_parser = filing_parser
        self.sp500 = sp500_companies
        # RAG modules loaded lazily only when needed
        self._rag_retriever = None
        self._document_processor = None

    @property
    def rag_retriever(self):
        """Lazy-load RAG retriever only when needed"""
        if self._rag_retriever is None:
            from app.services.rag.retriever import rag_retriever
            self._rag_retriever = rag_retriever
        return self._rag_retriever

    @property
    def document_processor(self):
        """Lazy-load document processor only when needed"""
        if self._document_processor is None:
            from app.services.rag.document_processor import document_processor
            self._document_processor = document_processor
        return self._document_processor

    def ingest_company_filings(
        self,
        ticker: str,
        num_filings: int = 5,
        form_types: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Ingest filings for a single company

        Args:
            ticker: Stock ticker symbol
            num_filings: Number of most recent filings to ingest
            form_types: List of form types (default: ['10-K', '10-Q'])

        Returns:
            Dictionary with ingestion results
        """
        if form_types is None:
            form_types = ['10-K', '10-Q']

        try:
            logger.info(f"Starting ingestion for {ticker}")

            # Get CIK for ticker
            cik = self.edgar_client.get_cik_by_ticker(ticker)
            if not cik:
                logger.error(f"Could not find CIK for ticker {ticker}")
                return {
                    'success': False,
                    'ticker': ticker,
                    'error': 'CIK not found'
                }

            # Update CIK in company database
            self.sp500.update_cik(ticker, cik)

            logger.info(f"Found CIK {cik} for {ticker}")

            # Get filings
            filings = self.edgar_client.get_company_filings(
                cik,
                form_types=form_types
            )

            if not filings:
                logger.warning(f"No filings found for {ticker}")
                return {
                    'success': True,
                    'ticker': ticker,
                    'cik': cik,
                    'filings_processed': 0,
                    'chunks_created': 0,
                    'message': 'No filings found'
                }

            # Limit number of filings
            filings = filings[:num_filings]

            # Process each filing
            total_chunks = 0
            processed_filings = []

            for filing in filings:
                result = self._process_filing(ticker, cik, filing)
                if result['success']:
                    total_chunks += result['chunks_created']
                    processed_filings.append({
                        'form_type': filing['form_type'],
                        'filing_date': filing['filing_date'],
                        'chunks': result['chunks_created']
                    })

            logger.info(
                f"Completed ingestion for {ticker}: "
                f"{len(processed_filings)} filings, {total_chunks} chunks"
            )

            return {
                'success': True,
                'ticker': ticker,
                'cik': cik,
                'filings_processed': len(processed_filings),
                'chunks_created': total_chunks,
                'filings': processed_filings
            }

        except Exception as e:
            logger.error(f"Error ingesting company filings: {e}")
            return {
                'success': False,
                'ticker': ticker,
                'error': str(e)
            }

    def _process_filing(
        self,
        ticker: str,
        cik: str,
        filing: Dict
    ) -> Dict[str, any]:
        """
        Process a single filing

        Args:
            ticker: Stock ticker
            cik: Company CIK
            filing: Filing dictionary

        Returns:
            Processing result
        """
        try:
            logger.info(
                f"Processing {filing['form_type']} for {ticker} "
                f"dated {filing['filing_date']}"
            )

            # Download filing document
            html_content = self.edgar_client.download_filing_text(
                cik,
                filing['accession_number'],
                filing['primary_document']
            )

            if not html_content:
                logger.error("Failed to download filing")
                return {'success': False, 'error': 'Download failed'}

            # Parse filing
            parsed = self.filing_parser.parse_filing(
                html_content,
                filing_type=filing['form_type']
            )

            # Get key sections text (focus on important sections)
            key_sections = ['md_and_a', 'risk_factors', 'business', 'financial_information']
            text_to_ingest = self.filing_parser.get_key_sections_text(
                parsed['sections'],
                section_names=key_sections
            )

            # Fallback to full text if no sections extracted
            if not text_to_ingest:
                text_to_ingest = parsed['full_text'][:500000]  # Limit to 500k chars

            # Create metadata
            metadata = {
                'ticker': ticker,
                'company': self.sp500.get_company_by_ticker(ticker).get('company_name', ticker),
                'cik': cik,
                'document_type': filing['form_type'],
                'filing_date': filing['filing_date'],
                'report_date': filing['report_date'],
                'accession_number': filing['accession_number'],
                'source': 'SEC EDGAR',
                'ingestion_date': datetime.now().isoformat(),
            }

            # Chunk the text
            chunks = self.document_processor.chunk_text(text_to_ingest, metadata)

            # Extract texts and metadata
            texts = [chunk['text'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]

            # Add to vector store
            chunk_ids = self.rag_retriever.vector_store.add_documents(texts, metadatas)

            logger.info(f"Created {len(chunk_ids)} chunks from {filing['form_type']}")

            return {
                'success': True,
                'chunks_created': len(chunk_ids),
                'text_length': len(text_to_ingest)
            }

        except Exception as e:
            logger.error(f"Error processing filing: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _process_filing_with_storage(
        self,
        ticker: str,
        cik: str,
        filing: Dict,
        save_to_disk: bool = True,
        ingest_to_rag: bool = False
    ) -> Dict[str, any]:
        """
        Process filing with file storage option (for MVP)

        Args:
            ticker: Stock ticker
            cik: Company CIK
            filing: Filing dictionary
            save_to_disk: Whether to save files to disk
            ingest_to_rag: Whether to also ingest to RAG

        Returns:
            Processing result
        """
        try:
            logger.info(
                f"Processing {filing['form_type']} for {ticker} "
                f"dated {filing['filing_date']}"
            )

            # Download filing document
            html_content = self.edgar_client.download_filing_text(
                cik,
                filing['accession_number'],
                filing['primary_document']
            )

            if not html_content:
                logger.error("Failed to download filing")
                return {'success': False, 'error': 'Download failed'}

            # Parse filing
            parsed = self.filing_parser.parse_filing(
                html_content,
                filing_type=filing['form_type']
            )

            # Get key sections text
            key_sections = ['md_and_a', 'risk_factors', 'business', 'financial_information']
            text_to_ingest = self.filing_parser.get_key_sections_text(
                parsed['sections'],
                section_names=key_sections
            )

            # Fallback to full text if no sections extracted
            if not text_to_ingest:
                text_to_ingest = parsed['full_text'][:500000]

            # Save to disk if requested (MVP approach)
            if save_to_disk:
                file_storage.save_edgar_filing(
                    ticker=ticker,
                    form_type=filing['form_type'],
                    date=filing['filing_date'],
                    raw_content=html_content,
                    parsed_text=text_to_ingest,
                    metadata={
                        'cik': cik,
                        'accession_number': filing['accession_number'],
                        'report_date': filing['report_date']
                    }
                )

            # Optionally ingest to RAG
            chunks_created = 0
            if ingest_to_rag:
                metadata = {
                    'ticker': ticker,
                    'company': self.sp500.get_company_by_ticker(ticker).get('company_name', ticker),
                    'cik': cik,
                    'document_type': filing['form_type'],
                    'filing_date': filing['filing_date'],
                    'report_date': filing['report_date'],
                    'accession_number': filing['accession_number'],
                    'source': 'SEC EDGAR',
                    'ingestion_date': datetime.now().isoformat(),
                }

                chunks = self.document_processor.chunk_text(text_to_ingest, metadata)
                texts = [chunk['text'] for chunk in chunks]
                metadatas = [chunk['metadata'] for chunk in chunks]
                chunk_ids = self.rag_retriever.vector_store.add_documents(texts, metadatas)
                chunks_created = len(chunk_ids)

            logger.info(
                f"Processed {filing['form_type']} for {ticker}: "
                f"saved={save_to_disk}, chunks={chunks_created}"
            )

            return {
                'success': True,
                'chunks_created': chunks_created,
                'text_length': len(text_to_ingest),
                'saved_to_disk': save_to_disk
            }

        except Exception as e:
            logger.error(f"Error processing filing: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def batch_ingest_companies(
        self,
        tickers: List[str],
        num_filings: int = 3
    ) -> Dict[str, any]:
        """
        Ingest filings for multiple companies

        Args:
            tickers: List of stock tickers
            num_filings: Number of filings per company

        Returns:
            Batch ingestion results
        """
        results = []
        total_filings = 0
        total_chunks = 0

        logger.info(f"Starting batch ingestion for {len(tickers)} companies")

        for ticker in tickers:
            result = self.ingest_company_filings(ticker, num_filings)
            results.append(result)

            if result['success']:
                total_filings += result['filings_processed']
                total_chunks += result['chunks_created']

        logger.info(
            f"Batch ingestion complete: "
            f"{len(tickers)} companies, {total_filings} filings, {total_chunks} chunks"
        )

        return {
            'success': True,
            'companies_processed': len(tickers),
            'total_filings': total_filings,
            'total_chunks': total_chunks,
            'results': results
        }

    def ingest_sp500_sample(
        self,
        sample_size: int = 10,
        num_filings: int = 2
    ) -> Dict[str, any]:
        """
        Ingest a sample of S&P 500 companies

        Args:
            sample_size: Number of companies to sample
            num_filings: Number of filings per company

        Returns:
            Ingestion results
        """
        tickers = self.sp500.get_tickers()[:sample_size]
        return self.batch_ingest_companies(tickers, num_filings)


# Singleton instance
edgar_ingestion = EDGARIngestion()
