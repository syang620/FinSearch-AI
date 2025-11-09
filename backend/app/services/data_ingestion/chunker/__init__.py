"""
Semantic Chunking Module for FinSearch AI

This module provides utilities for chunking parsed documents (EDGAR filings and
earnings transcripts) into optimized segments for RAG embedding and retrieval.

Key Features:
- Section-aware chunking for EDGAR filings (respects Item boundaries)
- Speaker/phase-aware chunking for earnings transcripts (respects Q&A exchanges)
- Token-constrained packing with configurable overlap
- Preserves rich metadata from source documents

Example:
    >>> from chunker import chunk_document, load_config
    >>> config = load_config('configs/chunking.yaml')
    >>> chunks = chunk_document('data_parsed/AAPL/edgar/AAPL_10K_2024.jsonl', config)
    >>> len(chunks)
    145
"""

from .tokenize import get_tokenizer, count_tokens
from .sections import split_filing_sections, split_transcript
from .pack import paragraph_units, sentence_units, pack_units
from .window import window_with_overlap
from .runner import chunk_document, load_config, save_chunked_output, build_chunk

__version__ = "1.0.0"
__all__ = [
    "get_tokenizer",
    "count_tokens",
    "split_filing_sections",
    "split_transcript",
    "paragraph_units",
    "sentence_units",
    "pack_units",
    "window_with_overlap",
    "chunk_document",
    "load_config",
    "save_chunked_output",
    "build_chunk",
]
