"""
Document chunking strategies for RAG pipeline.
Consolidates multiple chunking approaches into a single configurable class.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import re
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np


@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]
    chunk_id: str

    def __len__(self):
        return len(self.text)


class DocumentChunker:
    """
    Unified document chunking with multiple strategies.
    Consolidates pack.py, window.py, sections.py, etc. into one interface.
    """

    def __init__(
        self,
        strategy: str = 'semantic',
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        tokenizer: str = 'cl100k_base',
        min_chunk_size: int = 100
    ):
        """
        Initialize the chunker with configurable strategy.

        Args:
            strategy: Chunking strategy ('semantic', 'window', 'sentence', 'section')
            chunk_size: Target size for chunks (in tokens)
            chunk_overlap: Overlap between chunks (in tokens)
            tokenizer: Tokenizer to use for counting tokens
            min_chunk_size: Minimum size for a valid chunk
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding(tokenizer)

        # Load semantic model if needed
        if strategy == 'semantic':
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk document based on selected strategy.

        Args:
            text: Document text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        metadata = metadata or {}

        if self.strategy == 'semantic':
            return self._semantic_chunk(text, metadata)
        elif self.strategy == 'window':
            return self._window_chunk(text, metadata)
        elif self.strategy == 'sentence':
            return self._sentence_chunk(text, metadata)
        elif self.strategy == 'section':
            return self._section_chunk(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def _window_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Simple sliding window chunking.
        Replaces window.py functionality.
        """
        chunks = []
        tokens = self.tokenizer.encode(text)

        start = 0
        chunk_num = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))

            # Decode chunk text
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk = Chunk(
                    text=chunk_text,
                    start_idx=start,
                    end_idx=end,
                    metadata={**metadata, 'chunk_num': chunk_num},
                    chunk_id=f"{metadata.get('doc_id', 'doc')}_{chunk_num}"
                )
                chunks.append(chunk)
                chunk_num += 1

            # Move window
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _sentence_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk by sentences, respecting chunk size limits.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_num = 0

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))

            # If adding this sentence exceeds chunk size, create chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = Chunk(
                    text=chunk_text,
                    start_idx=text.find(chunk_text),
                    end_idx=text.find(chunk_text) + len(chunk_text),
                    metadata={**metadata, 'chunk_num': chunk_num},
                    chunk_id=f"{metadata.get('doc_id', 'doc')}_{chunk_num}"
                )
                chunks.append(chunk)
                chunk_num += 1

                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = []
                    overlap_tokens = 0
                    for sent in reversed(current_chunk):
                        sent_tok = len(self.tokenizer.encode(sent))
                        if overlap_tokens + sent_tok <= self.chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_tokens += sent_tok
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk = Chunk(
                    text=chunk_text,
                    start_idx=text.find(chunk_text),
                    end_idx=text.find(chunk_text) + len(chunk_text),
                    metadata={**metadata, 'chunk_num': chunk_num},
                    chunk_id=f"{metadata.get('doc_id', 'doc')}_{chunk_num}"
                )
                chunks.append(chunk)

        return chunks

    def _section_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk by document sections (headers, paragraphs).
        Replaces sections.py functionality.
        """
        # Identify sections by headers or double newlines
        section_pattern = r'(?:^|\n\n)(?:#+\s*|[A-Z][A-Z\s]+:)'
        sections = re.split(section_pattern, text)

        chunks = []
        chunk_num = 0

        for section in sections:
            if not section.strip():
                continue

            # If section is too large, use window chunking on it
            section_tokens = len(self.tokenizer.encode(section))

            if section_tokens > self.chunk_size:
                # Recursively chunk large sections
                sub_chunker = DocumentChunker(
                    strategy='window',
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    tokenizer='cl100k_base',
                    min_chunk_size=self.min_chunk_size
                )
                sub_chunks = sub_chunker.chunk(section, metadata)

                # Renumber chunk IDs
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = f"{metadata.get('doc_id', 'doc')}_{chunk_num}"
                    sub_chunk.metadata['chunk_num'] = chunk_num
                    chunks.append(sub_chunk)
                    chunk_num += 1
            else:
                # Section fits in single chunk
                if len(section.strip()) >= self.min_chunk_size:
                    chunk = Chunk(
                        text=section,
                        start_idx=text.find(section),
                        end_idx=text.find(section) + len(section),
                        metadata={**metadata, 'chunk_num': chunk_num},
                        chunk_id=f"{metadata.get('doc_id', 'doc')}_{chunk_num}"
                    )
                    chunks.append(chunk)
                    chunk_num += 1

        return chunks

    def _semantic_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Semantic chunking using sentence embeddings.
        Groups similar sentences together.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if not sentences:
            return []

        # Get embeddings for all sentences
        embeddings = self.semantic_model.encode(sentences)

        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(sim)

        # Find breakpoints where similarity is low
        if similarities:
            threshold = np.percentile(similarities, 25)  # Bottom 25% similarity
            breakpoints = [0]
            for i, sim in enumerate(similarities):
                if sim < threshold:
                    breakpoints.append(i + 1)
            breakpoints.append(len(sentences))
        else:
            breakpoints = [0, len(sentences)]

        # Create chunks from breakpoints
        chunks = []
        chunk_num = 0

        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)

            # Check token count
            tokens = self.tokenizer.encode(chunk_text)

            # If chunk is too large, split it
            if len(tokens) > self.chunk_size:
                # Use window chunking as fallback
                sub_chunker = DocumentChunker(
                    strategy='window',
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    tokenizer='cl100k_base',
                    min_chunk_size=self.min_chunk_size
                )
                sub_chunks = sub_chunker.chunk(chunk_text, metadata)

                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = f"{metadata.get('doc_id', 'doc')}_{chunk_num}"
                    sub_chunk.metadata['chunk_num'] = chunk_num
                    chunks.append(sub_chunk)
                    chunk_num += 1
            else:
                # Chunk is good size
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunk = Chunk(
                        text=chunk_text,
                        start_idx=text.find(chunk_text),
                        end_idx=text.find(chunk_text) + len(chunk_text),
                        metadata={**metadata, 'chunk_num': chunk_num},
                        chunk_id=f"{metadata.get('doc_id', 'doc')}_{chunk_num}"
                    )
                    chunks.append(chunk)
                    chunk_num += 1

        return chunks

    def validate_chunks(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Validate chunks and return statistics.
        Replaces validate.py functionality.
        """
        if not chunks:
            return {
                'valid': False,
                'num_chunks': 0,
                'error': 'No chunks generated'
            }

        stats = {
            'valid': True,
            'num_chunks': len(chunks),
            'avg_size': np.mean([len(c.text) for c in chunks]),
            'min_size': min(len(c.text) for c in chunks),
            'max_size': max(len(c.text) for c in chunks),
            'avg_tokens': np.mean([len(self.tokenizer.encode(c.text)) for c in chunks]),
            'total_text_length': sum(len(c.text) for c in chunks),
            'chunks_below_min': sum(1 for c in chunks if len(c.text) < self.min_chunk_size)
        }

        # Check for issues
        if stats['chunks_below_min'] > 0:
            stats['warning'] = f"{stats['chunks_below_min']} chunks below minimum size"

        return stats