from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from docx import Document
import pandas as pd
from pathlib import Path
import logging
import re
from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for RAG with section-aware and semantic chunking"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def process_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise

    def process_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            raise

    def process_excel(self, file_path: str) -> str:
        """Extract text from Excel file"""
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            text = ""
            for sheet_name, sheet_df in df.items():
                text += f"\n--- Sheet: {sheet_name} ---\n"
                text += sheet_df.to_string(index=False) + "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing Excel {file_path}: {e}")
            raise

    def process_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            raise

    def process_file(self, file_path: str) -> str:
        """
        Process a file based on its extension

        Args:
            file_path: Path to the file

        Returns:
            Extracted text content
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        processors = {
            '.pdf': self.process_pdf,
            '.docx': self.process_docx,
            '.doc': self.process_docx,
            '.xlsx': self.process_excel,
            '.xls': self.process_excel,
            '.txt': self.process_text,
        }

        processor = processors.get(extension)
        if processor is None:
            raise ValueError(f"Unsupported file type: {extension}")

        return processor(file_path)

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks using semantic and section-aware chunking

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
                     Can include 'document_type' and 'sections' for section-aware chunking

        Returns:
            List of chunk dictionaries with text and metadata
        """
        metadata = metadata or {}
        document_type = metadata.get('document_type', '')

        # Use section-aware chunking for EDGAR filings
        if document_type in ['10-K', '10-Q'] and 'sections' in metadata:
            return self._chunk_by_sections(text, metadata)

        # Use semantic chunking for all other documents
        return self._chunk_semantically(text, metadata)

    def _chunk_by_sections(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk EDGAR filings by detected sections for better semantic coherence

        Args:
            text: Filing text
            metadata: Metadata including sections information

        Returns:
            List of chunks based on sections
        """
        chunks = []
        sections = metadata.get('sections', {})

        if not sections:
            # Fallback to semantic chunking if no sections
            return self._chunk_semantically(text, metadata)

        for section_name, section_text in sections.items():
            # Create section-specific metadata
            section_metadata = metadata.copy()
            section_metadata['section_name'] = section_name
            section_metadata['section_type'] = self._categorize_section(section_name)

            # Chunk the section if it's too large
            if len(section_text) > self.chunk_size:
                section_chunks = self._chunk_semantically(section_text, section_metadata)
                chunks.extend(section_chunks)
            else:
                # Keep small sections as single chunks
                chunk_metadata = section_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": len(chunks),
                    "start_char": 0,
                    "end_char": len(section_text),
                })
                chunks.append({
                    "text": section_text,
                    "metadata": chunk_metadata
                })

        logger.info(f"Created {len(chunks)} section-aware chunks from {len(sections)} sections")
        return chunks

    def _chunk_semantically(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text semantically by respecting sentence and paragraph boundaries

        Args:
            text: Input text
            metadata: Optional metadata

        Returns:
            List of semantic chunks
        """
        chunks = []
        metadata = metadata or {}

        # Split by paragraphs first
        paragraphs = re.split(r'\n\n+', text)

        current_chunk = ""
        current_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": len(chunks),
                        "start_char": current_start,
                        "end_char": current_start + len(current_chunk),
                    })
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": chunk_metadata
                    })

                    # Start new chunk with overlap (last paragraph)
                    current_start += len(current_chunk) - self.chunk_overlap
                    current_chunk = para + "\n\n"
                else:
                    # Paragraph itself is too large, split by sentences
                    sentence_chunks = self._split_long_paragraph(para, metadata, len(chunks), current_start)
                    chunks.extend(sentence_chunks)
                    current_start += sum(len(c["text"]) for c in sentence_chunks)
                    current_chunk = ""
            else:
                # Add paragraph to current chunk
                current_chunk += para + "\n\n"

        # Add final chunk
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),
                "start_char": current_start,
                "end_char": current_start + len(current_chunk),
            })
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": chunk_metadata
            })

        logger.info(f"Created {len(chunks)} semantic chunks from text of length {len(text)}")
        return chunks

    def _split_long_paragraph(self, paragraph: str, metadata: Dict[str, Any],
                             start_chunk_idx: int, start_char: int) -> List[Dict[str, Any]]:
        """
        Split a long paragraph by sentences when it exceeds chunk size

        Args:
            paragraph: Long paragraph text
            metadata: Base metadata
            start_chunk_idx: Starting chunk index
            start_char: Starting character position

        Returns:
            List of chunks from the paragraph
        """
        chunks = []

        # Split into sentences (simple regex)
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)

        current_chunk = ""
        current_start = start_char

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": start_chunk_idx + len(chunks),
                        "start_char": current_start,
                        "end_char": current_start + len(current_chunk),
                    })
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": chunk_metadata
                    })
                    current_start += len(current_chunk)
                    current_chunk = sentence + " "
                else:
                    # Single sentence is too long, force split
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": start_chunk_idx + len(chunks),
                        "start_char": current_start,
                        "end_char": current_start + self.chunk_size,
                    })
                    chunks.append({
                        "text": sentence[:self.chunk_size],
                        "metadata": chunk_metadata
                    })
                    current_start += self.chunk_size
                    current_chunk = sentence[self.chunk_size:] + " " if len(sentence) > self.chunk_size else ""
            else:
                current_chunk += sentence + " "

        # Add remaining text
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": start_chunk_idx + len(chunks),
                "start_char": current_start,
                "end_char": current_start + len(current_chunk),
            })
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": chunk_metadata
            })

        return chunks

    def _categorize_section(self, section_name: str) -> str:
        """Categorize EDGAR section type for metadata"""
        section_types = {
            'business': 'business_overview',
            'risk_factors': 'risk_disclosure',
            'md_and_a': 'management_discussion',
            'financial_statements': 'financials',
            'financial_data': 'financials',
            'financial_information': 'financials',
            'legal_proceedings': 'legal',
            'properties': 'assets',
            'controls': 'governance',
            'quantitative_disclosures': 'risk_disclosure',
        }
        return section_types.get(section_name, 'other')

    def process_and_chunk_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a file and split into chunks

        Args:
            file_path: Path to the file
            metadata: Optional metadata for the document

        Returns:
            List of chunks with text and metadata
        """
        # Extract text
        text = self.process_file(file_path)

        # Add file info to metadata
        if metadata is None:
            metadata = {}

        path = Path(file_path)
        metadata.update({
            "filename": path.name,
            "file_type": path.suffix.lower(),
            "file_size": path.stat().st_size,
        })

        # Chunk the text
        chunks = self.chunk_text(text, metadata)

        return chunks


# Singleton instance
document_processor = DocumentProcessor()
