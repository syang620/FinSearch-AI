"""
Tokenization Utilities

Provides token counting and encoding/decoding functions for chunking.
Supports tiktoken (OpenAI models) with fallback options.
"""

from typing import Callable, Tuple
import logging

logger = logging.getLogger(__name__)


def get_tokenizer(name: str) -> Tuple[Callable[[str], list], Callable[[list], str]]:
    """
    Get encoder and decoder functions for the specified tokenizer.

    Args:
        name: Tokenizer name. Supported values:
            - "cl100k_base": OpenAI GPT-4, GPT-3.5-turbo (recommended)
            - "p50k_base": OpenAI older models (GPT-3)

    Returns:
        Tuple of (encode_fn, decode_fn) where:
            - encode_fn: str -> List[int] (text to token IDs)
            - decode_fn: List[int] -> str (token IDs to text)

    Raises:
        ImportError: If tiktoken is not installed
        ValueError: If tokenizer name is not supported

    Example:
        >>> enc, dec = get_tokenizer("cl100k_base")
        >>> tokens = enc("Hello world")
        >>> len(tokens)
        2
        >>> dec(tokens)
        'Hello world'
    """
    if name == "cl100k_base":
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return encoding.encode, encoding.decode
        except ImportError:
            logger.error("tiktoken not installed. Install with: pip install tiktoken")
            raise
        except Exception as e:
            logger.error(f"Failed to load tiktoken encoding '{name}': {e}")
            raise

    elif name == "p50k_base":
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("p50k_base")
            return encoding.encode, encoding.decode
        except ImportError:
            logger.error("tiktoken not installed. Install with: pip install tiktoken")
            raise

    else:
        raise ValueError(
            f"Unsupported tokenizer: {name}. "
            f"Supported: 'cl100k_base', 'p50k_base'"
        )


def count_tokens(text: str, enc: Callable[[str], list]) -> int:
    """
    Count number of tokens in text using the provided encoder.

    Args:
        text: Text to count tokens for
        enc: Encoder function (from get_tokenizer)

    Returns:
        Number of tokens in the text

    Example:
        >>> enc, _ = get_tokenizer("cl100k_base")
        >>> count_tokens("The quick brown fox", enc)
        4
    """
    if not text:
        return 0

    try:
        tokens = enc(text)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Error counting tokens, falling back to estimate: {e}")
        # Fallback: rough estimate of 1 token ≈ 4 characters
        return len(text) // 4


def compute_overlap_tokens(chunk_tokens: int, overlap_ratio: float) -> int:
    """
    Compute number of overlap tokens based on chunk size and overlap ratio.

    Args:
        chunk_tokens: Size of the chunk in tokens
        overlap_ratio: Overlap percentage (e.g., 0.15 for 15%)

    Returns:
        Number of tokens to overlap

    Example:
        >>> compute_overlap_tokens(600, 0.15)
        90
        >>> compute_overlap_tokens(800, 0.20)
        160
    """
    return int(chunk_tokens * overlap_ratio)


def estimate_tokens_for_units(units: list, enc: Callable[[str], list]) -> int:
    """
    Estimate total tokens for a list of text units.

    Args:
        units: List of text strings or dicts with 'text' key
        enc: Encoder function

    Returns:
        Total token count across all units

    Example:
        >>> enc, _ = get_tokenizer("cl100k_base")
        >>> units = ["Hello", "world", "!"]
        >>> estimate_tokens_for_units(units, enc)
        3
    """
    total = 0
    for unit in units:
        # Handle both string lists and dict lists
        text = unit if isinstance(unit, str) else unit.get('text', '')
        total += count_tokens(text, enc)
    return total
