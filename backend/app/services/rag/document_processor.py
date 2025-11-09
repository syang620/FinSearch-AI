from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from docx import Document
import pandas as pd
from pathlib import Path
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for RAG"""

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
        Split text into chunks with overlap

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size

            # Extract chunk
            chunk = text[start:end]

            # Create chunk metadata
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_index": len(chunks),
                "start_char": start,
                "end_char": min(end, text_length),
            })

            chunks.append({
                "text": chunk,
                "metadata": chunk_metadata
            })

            # Move to next chunk with overlap
            start = end - self.chunk_overlap

            # Prevent infinite loop
            if start >= text_length or self.chunk_overlap >= self.chunk_size:
                break

        logger.info(f"Created {len(chunks)} chunks from text of length {text_length}")
        return chunks

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
