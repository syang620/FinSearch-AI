"""
Unit Packing

Utilities for packing text units (paragraphs, sentences, utterances) into
chunks that target a specific token size.
"""

import re
from typing import List, Callable
import logging

from .tokenize import count_tokens

logger = logging.getLogger(__name__)


def paragraph_units(text: str) -> List[str]:
    """
    Split text into paragraph units (separated by double newlines).

    Args:
        text: Input text to split

    Returns:
        List of paragraph strings (whitespace stripped, empty paragraphs removed)

    Example:
        >>> text = "First paragraph.\\n\\nSecond paragraph.\\n\\n\\nThird paragraph."
        >>> paras = paragraph_units(text)
        >>> len(paras)
        3
        >>> paras[0]
        'First paragraph.'
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    logger.debug(f"Split text into {len(paras)} paragraphs")
    return paras


def sentence_units(text: str) -> List[str]:
    """
    Split text into sentence units.

    Args:
        text: Input text to split

    Returns:
        List of sentence strings (empty sentences removed)

    Note:
        Uses a simple regex pattern that splits on periods, exclamation marks,
        and question marks followed by whitespace. This is a heuristic and may
        not be perfect for all text.

    Example:
        >>> text = "First sentence. Second sentence! Third sentence?"
        >>> sents = sentence_units(text)
        >>> len(sents)
        3
        >>> sents[0]
        'First sentence.'
    """
    # Split on sentence-ending punctuation followed by whitespace
    # Handles: . ! ?
    sents = re.split(r'(?<=[\\.!?])\\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]

    logger.debug(f"Split text into {len(sents)} sentences")
    return sents


def pack_units(
    units: List[str],
    enc: Callable[[str], list],
    target: int,
    min_tokens: int
) -> List[str]:
    """
    Pack text units into chunks targeting a specific token size.

    Algorithm:
        1. Accumulate units while total tokens < target OR total tokens < min_tokens
        2. When adding the next unit would exceed target AND we're above min_tokens:
           - Finalize current chunk
           - Start new chunk with the current unit
        3. Always keep accumulating if below min_tokens (prevents tiny chunks)

    Args:
        units: List of text units (paragraphs, sentences, etc.)
        enc: Token encoder function
        target: Target chunk size in tokens
        min_tokens: Minimum chunk size in tokens (prevents very small chunks)

    Returns:
        List of packed chunk texts (units joined with spaces)

    Example:
        >>> units = ["Short.", "Also short.", "This is a longer sentence with more words."]
        >>> enc, _ = get_tokenizer("cl100k_base")
        >>> chunks = pack_units(units, enc, target=10, min_tokens=3)
        >>> len(chunks)
        2
        >>> chunks[0]
        'Short. Also short.'
    """
    if not units:
        return []

    chunks = []
    current_units = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit, enc)

        # Add unit if:
        # 1. We haven't reached target yet, OR
        # 2. We're still below minimum (prevent tiny chunks)
        if current_tokens + unit_tokens < target or current_tokens < min_tokens:
            current_units.append(unit)
            current_tokens += unit_tokens
        else:
            # Finalize current chunk
            if current_units:
                chunks.append(" ".join(current_units))
                logger.debug(f"Packed chunk: {current_tokens} tokens from {len(current_units)} units")

            # Start new chunk
            current_units = [unit]
            current_tokens = unit_tokens

    # Don't forget the last chunk
    if current_units:
        chunks.append(" ".join(current_units))
        logger.debug(f"Packed chunk: {current_tokens} tokens from {len(current_units)} units")

    logger.info(f"Packed {len(units)} units into {len(chunks)} chunks (target={target}, min={min_tokens})")
    return chunks


def merge_small_chunks(
    chunks: List[str],
    min_tokens: int,
    enc: Callable[[str], list]
) -> List[str]:
    """
    Merge chunks that are smaller than the minimum token size.

    Args:
        chunks: List of chunk texts
        min_tokens: Minimum acceptable chunk size
        enc: Token encoder function

    Returns:
        List of chunks with small chunks merged into adjacent ones

    Example:
        >>> chunks = ["Tiny", "Also tiny", "This is a longer chunk"]
        >>> enc, _ = get_tokenizer("cl100k_base")
        >>> merged = merge_small_chunks(chunks, min_tokens=5, enc=enc)
        >>> len(merged)
        2
        >>> merged[0]
        'Tiny Also tiny'
    """
    if not chunks:
        return []

    merged = []
    accumulator = []
    accumulator_tokens = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk, enc)

        if chunk_tokens >= min_tokens:
            # Chunk is large enough
            if accumulator:
                # Flush any accumulated small chunks
                merged.append(" ".join(accumulator))
                accumulator = []
                accumulator_tokens = 0

            merged.append(chunk)
        else:
            # Chunk is too small, accumulate
            accumulator.append(chunk)
            accumulator_tokens += chunk_tokens

            # If accumulator is now large enough, flush it
            if accumulator_tokens >= min_tokens:
                merged.append(" ".join(accumulator))
                accumulator = []
                accumulator_tokens = 0

    # Flush any remaining accumulated chunks
    if accumulator:
        if merged:
            # Merge with last chunk if possible
            merged[-1] = merged[-1] + " " + " ".join(accumulator)
        else:
            # No other chunks, keep the small one
            merged.append(" ".join(accumulator))

    logger.info(f"Merged {len(chunks)} chunks into {len(merged)} (min_tokens={min_tokens})")
    return merged


def combine_texts(units: List[dict]) -> str:
    """
    Combine text from a list of unit dictionaries.

    Args:
        units: List of dicts with 'text' key

    Returns:
        Combined text (joined with spaces)

    Example:
        >>> units = [{"text": "Hello"}, {"text": "world"}]
        >>> combine_texts(units)
        'Hello world'
    """
    texts = [u.get('text', '') for u in units]
    return " ".join(t for t in texts if t)
