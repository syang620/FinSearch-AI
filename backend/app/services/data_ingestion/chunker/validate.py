"""
Chunk Validation

Utilities for validating chunked output against schema requirements.
"""

from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# Required fields for all chunks
REQUIRED_FIELDS = {
    "chunk_id",
    "doc_id",
    "ticker",
    "company",
    "doc_type",
    "fiscal_year",
    "period",
    "text",
    "chunk_tokens",
    "chunk_type",
    "chunked_at"
}

# EDGAR-specific fields (should be non-null for 10-K/10-Q)
EDGAR_FIELDS = {"section_id", "section_title"}

# Transcript-specific fields (should be non-null for earnings_transcript)
TRANSCRIPT_FIELDS = {"phase", "speaker", "speaker_role"}

# Valid doc types
VALID_DOC_TYPES = {"10-K", "10-Q", "earnings_transcript"}

# Valid chunk types
VALID_CHUNK_TYPES = {
    "section_packed",
    "windowed_overlap",
    "prepared_packed",
    "qa_exchange",
    "packed"
}


def validate_chunk_metadata(chunk: Dict) -> Tuple[bool, List[str]]:
    """
    Validate chunk metadata against schema requirements.

    Args:
        chunk: Chunk dictionary to validate

    Returns:
        Tuple of (is_valid, error_messages)

    Example:
        >>> chunk = {...}
        >>> is_valid, errors = validate_chunk_metadata(chunk)
        >>> if not is_valid:
        ...     for err in errors:
        ...         print(f"ERROR: {err}")
    """
    errors = []

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in chunk:
            errors.append(f"Missing required field: {field}")
        elif chunk[field] is None:
            errors.append(f"Required field is null: {field}")

    # Check doc_type validity
    doc_type = chunk.get("doc_type")
    if doc_type and doc_type not in VALID_DOC_TYPES:
        errors.append(f"Invalid doc_type: {doc_type}")

    # Check chunk_type validity
    chunk_type = chunk.get("chunk_type")
    if chunk_type and chunk_type not in VALID_CHUNK_TYPES:
        errors.append(f"Invalid chunk_type: {chunk_type}")

    # Check doc-type-specific fields
    if doc_type in ["10-K", "10-Q"]:
        # EDGAR documents should have section info
        for field in EDGAR_FIELDS:
            if field not in chunk:
                errors.append(f"Missing EDGAR field: {field}")
            # Note: section_id/title can be null for unsectioned content

        # Transcript fields should be null
        for field in TRANSCRIPT_FIELDS:
            if chunk.get(field) is not None:
                errors.append(f"EDGAR document should not have {field}")

    elif doc_type == "earnings_transcript":
        # Transcript should have phase/speaker info
        for field in TRANSCRIPT_FIELDS:
            if field not in chunk:
                errors.append(f"Missing transcript field: {field}")
            # Note: speaker can be null for operator/unknown

        # EDGAR fields should be null
        for field in EDGAR_FIELDS:
            if chunk.get(field) is not None:
                errors.append(f"Transcript should not have {field}")

    # Check text content
    text = chunk.get("text", "")
    if not text or not text.strip():
        errors.append("Chunk text is empty")

    # Check chunk_id format
    chunk_id = chunk.get("chunk_id", "")
    if not chunk_id.endswith("_chunk_" + chunk_id.split("_chunk_")[-1] if "_chunk_" in chunk_id else ""):
        errors.append(f"Invalid chunk_id format: {chunk_id}")

    return (len(errors) == 0, errors)


def validate_chunk_tokens(
    chunk: Dict,
    min_tokens: int = 200,
    max_tokens: int = 800
) -> Tuple[bool, List[str]]:
    """
    Validate chunk token constraints.

    Args:
        chunk: Chunk dictionary
        min_tokens: Minimum acceptable tokens
        max_tokens: Maximum acceptable tokens

    Returns:
        Tuple of (is_valid, error_messages)

    Example:
        >>> chunk = {"chunk_tokens": 650, ...}
        >>> is_valid, errors = validate_chunk_tokens(chunk, min_tokens=200, max_tokens=800)
        >>> is_valid
        True
    """
    errors = []

    chunk_tokens = chunk.get("chunk_tokens")

    if chunk_tokens is None:
        errors.append("chunk_tokens field is missing or null")
        return (False, errors)

    # Allow some flexibility for very small chunks (e.g., last chunk in document)
    # But warn if significantly below minimum
    if chunk_tokens < min_tokens * 0.5:
        errors.append(
            f"Chunk very small: {chunk_tokens} tokens "
            f"(min={min_tokens}, half_min={min_tokens * 0.5})"
        )

    # Hard constraint on maximum
    if chunk_tokens > max_tokens:
        errors.append(
            f"Chunk exceeds maximum: {chunk_tokens} tokens (max={max_tokens})"
        )

    return (len(errors) == 0, errors)


def validate_overlap_consistency(chunk: Dict) -> Tuple[bool, List[str]]:
    """
    Validate overlap metadata consistency.

    Args:
        chunk: Chunk dictionary

    Returns:
        Tuple of (is_valid, error_messages)

    Example:
        >>> chunk = {"overlap_with_prev": 120, "overlap_with_next": 120, ...}
        >>> is_valid, errors = validate_overlap_consistency(chunk)
    """
    errors = []

    overlap_prev = chunk.get("overlap_with_prev", 0)
    overlap_next = chunk.get("overlap_with_next", 0)

    # Overlaps should be non-negative
    if overlap_prev < 0:
        errors.append(f"Invalid overlap_with_prev: {overlap_prev} (must be >= 0)")

    if overlap_next < 0:
        errors.append(f"Invalid overlap_with_next: {overlap_next} (must be >= 0)")

    # If chunk has windowed_overlap type, it should have overlaps
    chunk_type = chunk.get("chunk_type")
    if chunk_type == "windowed_overlap":
        if overlap_prev == 0 and overlap_next == 0:
            errors.append(
                "Windowed chunk should have overlap with prev or next chunk"
            )

    return (len(errors) == 0, errors)


def validate_chunks(
    chunks: List[Dict],
    min_tokens: int = 200,
    max_tokens: int = 800,
    verbose: bool = False
) -> Dict:
    """
    Validate list of chunks and return validation report.

    Args:
        chunks: List of chunk dictionaries
        min_tokens: Minimum acceptable tokens per chunk
        max_tokens: Maximum acceptable tokens per chunk
        verbose: If True, log all errors; if False, only summary

    Returns:
        Validation report dictionary with:
            - total_chunks: int
            - valid_chunks: int
            - invalid_chunks: int
            - errors_by_chunk: Dict[int, List[str]]
            - error_summary: Dict[str, int]

    Example:
        >>> chunks = [chunk1, chunk2, chunk3]
        >>> report = validate_chunks(chunks, min_tokens=200, max_tokens=800)
        >>> print(f"Valid: {report['valid_chunks']}/{report['total_chunks']}")
    """
    total = len(chunks)
    valid = 0
    errors_by_chunk = {}
    error_types = {}

    for i, chunk in enumerate(chunks):
        chunk_errors = []

        # Validate metadata
        meta_valid, meta_errors = validate_chunk_metadata(chunk)
        if not meta_valid:
            chunk_errors.extend(meta_errors)

        # Validate tokens
        token_valid, token_errors = validate_chunk_tokens(chunk, min_tokens, max_tokens)
        if not token_valid:
            chunk_errors.extend(token_errors)

        # Validate overlap consistency
        overlap_valid, overlap_errors = validate_overlap_consistency(chunk)
        if not overlap_valid:
            chunk_errors.extend(overlap_errors)

        # Track results
        if chunk_errors:
            errors_by_chunk[i] = chunk_errors

            # Count error types
            for err in chunk_errors:
                error_type = err.split(":")[0]  # Get prefix before colon
                error_types[error_type] = error_types.get(error_type, 0) + 1

            if verbose:
                logger.warning(
                    f"Chunk {i} ({chunk.get('chunk_id', 'UNKNOWN')}): "
                    f"{len(chunk_errors)} errors"
                )
                for err in chunk_errors:
                    logger.warning(f"  - {err}")
        else:
            valid += 1

    invalid = total - valid

    logger.info(
        f"Validation complete: {valid}/{total} valid chunks "
        f"({invalid} invalid)"
    )

    if invalid > 0 and not verbose:
        logger.warning(
            f"Found {invalid} invalid chunks. "
            f"Re-run with verbose=True to see all errors."
        )

    return {
        "total_chunks": total,
        "valid_chunks": valid,
        "invalid_chunks": invalid,
        "errors_by_chunk": errors_by_chunk,
        "error_summary": error_types
    }


def print_validation_report(report: Dict):
    """
    Print human-readable validation report.

    Args:
        report: Validation report from validate_chunks()

    Example:
        >>> report = validate_chunks(chunks)
        >>> print_validation_report(report)
    """
    print("\n" + "="*60)
    print("CHUNK VALIDATION REPORT")
    print("="*60)

    print(f"\nTotal chunks: {report['total_chunks']}")
    print(f"Valid chunks: {report['valid_chunks']} ✓")
    print(f"Invalid chunks: {report['invalid_chunks']} ✗")

    if report['invalid_chunks'] > 0:
        print(f"\nError Summary:")
        for error_type, count in sorted(
            report['error_summary'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  - {error_type}: {count}")

        print(f"\nInvalid chunk indices: {sorted(report['errors_by_chunk'].keys())[:10]}")
        if len(report['errors_by_chunk']) > 10:
            print(f"  ... and {len(report['errors_by_chunk']) - 10} more")

    print("="*60 + "\n")
