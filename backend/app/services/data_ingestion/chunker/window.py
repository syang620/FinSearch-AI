"""
Overlap Windowing

Utilities for applying sliding window with overlap to chunks that exceed
the maximum token limit.
"""

from typing import List, Tuple, Callable
import logging

from .tokenize import count_tokens

logger = logging.getLogger(__name__)


def window_with_overlap(
    long_text: str,
    enc: Callable[[str], list],
    dec: Callable[[list], str],
    max_tokens: int,
    overlap_ratio: float
) -> List[Tuple[int, int]]:
    """
    Apply sliding window with overlap to text exceeding max_tokens.

    Algorithm:
        1. Encode text to tokens
        2. Create overlapping windows of size max_tokens
        3. Step size = max_tokens * (1 - overlap_ratio)
        4. Stop when we reach the end of the text

    Args:
        long_text: Text to window
        enc: Token encoder function
        dec: Token decoder function
        max_tokens: Maximum tokens per window
        overlap_ratio: Overlap percentage (e.g., 0.15 for 15% overlap)

    Returns:
        List of (start_token_idx, end_token_idx) tuples representing window spans

    Example:
        >>> enc, dec = get_tokenizer("cl100k_base")
        >>> text = "..." # Some long text (1500 tokens)
        >>> spans = window_with_overlap(text, enc, dec, max_tokens=800, overlap_ratio=0.15)
        >>> spans
        [(0, 800), (680, 1480), (1360, 1500)]
        # Windows overlap by 120 tokens (800 * 0.15)
    """
    # Encode text to tokens
    tokens = enc(long_text)
    n = len(tokens)

    # Calculate step size (how far to advance for each window)
    # step = max_tokens * (1 - overlap_ratio)
    # Example: max_tokens=800, overlap=0.15 → step=680, overlap=120 tokens
    step = int(max_tokens * (1 - overlap_ratio))

    if step < 1:
        step = 1  # Prevent infinite loop

    windows = []

    # Create overlapping windows
    for start in range(0, n, step):
        end = min(start + max_tokens, n)

        # Skip if window is too small (edge case)
        if end - start < 1:
            break

        windows.append((start, end))

        # Stop if we've reached the end
        if end == n:
            break

    logger.debug(
        f"Created {len(windows)} windows for {n} tokens "
        f"(max={max_tokens}, overlap={overlap_ratio:.0%}, step={step})"
    )

    return windows


def extract_window_text(
    long_text: str,
    span: Tuple[int, int],
    enc: Callable[[str], list],
    dec: Callable[[list], str]
) -> str:
    """
    Extract text for a specific window span.

    Args:
        long_text: Full text
        span: (start_token_idx, end_token_idx) tuple
        enc: Token encoder function
        dec: Token decoder function

    Returns:
        Text corresponding to the token span

    Example:
        >>> enc, dec = get_tokenizer("cl100k_base")
        >>> text = "The quick brown fox jumps over the lazy dog"
        >>> extract_window_text(text, (0, 5), enc, dec)
        'The quick brown fox jumps'
    """
    start, end = span

    # Encode text
    tokens = enc(long_text)

    # Extract token span
    token_span = tokens[start:end]

    # Decode back to text
    return dec(token_span)


def split_large_chunks(
    chunks: List[str],
    max_tokens: int,
    overlap_ratio: float,
    enc: Callable[[str], list],
    dec: Callable[[list], str]
) -> List[Tuple[str, int, int]]:
    """
    Split chunks that exceed max_tokens using windowing with overlap.

    Args:
        chunks: List of chunk texts
        max_tokens: Maximum tokens per chunk
        overlap_ratio: Overlap percentage for windowing
        enc: Token encoder function
        dec: Token decoder function

    Returns:
        List of (chunk_text, start_token, end_token) tuples

    Example:
        >>> chunks = ["Short chunk", "This is a very long chunk..."]  # Second is 1200 tokens
        >>> enc, dec = get_tokenizer("cl100k_base")
        >>> result = split_large_chunks(chunks, max_tokens=800, overlap_ratio=0.15, enc=enc, dec=dec)
        >>> len(result)
        3  # First chunk unchanged, second chunk split into 2 windows
    """
    result = []

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk, enc)

        if chunk_tokens <= max_tokens:
            # Chunk is within limit, keep as is
            result.append((chunk, 0, chunk_tokens))
        else:
            # Chunk exceeds limit, apply windowing
            logger.debug(f"Splitting chunk with {chunk_tokens} tokens (max={max_tokens})")

            windows = window_with_overlap(chunk, enc, dec, max_tokens, overlap_ratio)

            for start, end in windows:
                window_text = extract_window_text(chunk, (start, end), enc, dec)
                result.append((window_text, start, end))

            logger.debug(f"Split into {len(windows)} windows")

    logger.info(f"Processed {len(chunks)} chunks into {len(result)} final chunks")
    return result


def compute_overlap_spans(
    windows: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Compute overlap token counts between consecutive windows.

    Args:
        windows: List of (start, end) token index tuples

    Returns:
        List of (overlap_with_prev, overlap_with_next) tuples

    Example:
        >>> windows = [(0, 800), (680, 1480), (1360, 1500)]
        >>> overlaps = compute_overlap_spans(windows)
        >>> overlaps
        [(0, 120), (120, 120), (120, 0)]
        # First window: no prev overlap, 120 tokens overlap with next
        # Middle window: 120 tokens overlap with prev and next
        # Last window: 120 tokens overlap with prev, no next overlap
    """
    overlaps = []

    for i, (start, end) in enumerate(windows):
        # Overlap with previous window
        if i > 0:
            prev_start, prev_end = windows[i - 1]
            overlap_prev = max(0, prev_end - start)
        else:
            overlap_prev = 0

        # Overlap with next window
        if i < len(windows) - 1:
            next_start, next_end = windows[i + 1]
            overlap_next = max(0, end - next_start)
        else:
            overlap_next = 0

        overlaps.append((overlap_prev, overlap_next))

    return overlaps
