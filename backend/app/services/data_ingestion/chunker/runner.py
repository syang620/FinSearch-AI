"""
Chunking Orchestration

Main entry point for document chunking. Coordinates all chunking strategies
and metadata building.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Callable
from datetime import datetime
import logging

from .tokenize import get_tokenizer, count_tokens
from .sections import group_by_exchange
from .pack import paragraph_units, pack_units, combine_texts
from .window import window_with_overlap, extract_window_text, compute_overlap_spans

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to chunking.yaml file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
            raise


def build_chunk(
    text: str,
    source_units: List[Dict],
    chunk_index: int,
    meta: Dict,
    start_token: int = 0,
    end_token: int = 0
) -> Dict:
    """
    Build complete chunk metadata dictionary.

    Attaches all required metadata fields:
    {ticker, doc_type, period, fiscal_year, section_id, section_title,
     speaker, source_uri, chunk_id, start_token, end_token, ...}

    Args:
        text: Chunk text content
        source_units: List of source paragraph/utterance dicts
        chunk_index: Index of this chunk in document
        meta: Additional metadata dict with:
            - section_id, section_title (for EDGAR)
            - speaker, speaker_role, phase (for transcripts)
            - chunk_type (e.g., "section_packed", "windowed_overlap")
        start_token: Start token offset (for windowed chunks)
        end_token: End token offset (for windowed chunks)

    Returns:
        Complete chunk dictionary with all metadata

    Example:
        >>> chunk = build_chunk(
        ...     text="Risk factors include...",
        ...     source_units=[para1, para2],
        ...     chunk_index=5,
        ...     meta={"section_id": "Item 1A", "chunk_type": "section_packed"}
        ... )
        >>> chunk['chunk_id']
        'AAPL_10K_2024_chunk_0005'
    """
    # Inherit from first source unit
    base = source_units[0] if source_units else {}

    # Calculate chunk tokens if not provided
    if end_token:
        chunk_tokens = end_token - start_token
    else:
        # Need encoder from meta
        enc = meta.get('enc')
        if enc:
            chunk_tokens = count_tokens(text, enc)
        else:
            chunk_tokens = len(text) // 4  # Rough estimate

    return {
        # Core identifiers
        "chunk_id": f"{base.get('doc_id', 'UNKNOWN')}_chunk_{chunk_index:04d}",
        "doc_id": base.get("doc_id"),

        # Source tracking
        "source_units": [u.get("chunk_id") for u in source_units],
        "source_uri": base.get("source_file"),

        # Company & period
        "ticker": base.get("ticker"),
        "company": base.get("company"),
        "doc_type": base.get("doc_type"),
        "fiscal_year": base.get("fiscal_year"),
        "quarter": base.get("quarter"),
        "period": base.get("period"),

        # Structure (from meta or base)
        "section_id": meta.get("section_id") or base.get("section_id"),
        "section_title": meta.get("section_title") or base.get("section_title"),
        "speaker": meta.get("speaker") or base.get("speaker_name"),
        "speaker_role": meta.get("speaker_role") or base.get("speaker_role"),
        "phase": meta.get("phase") or base.get("phase"),

        # Chunk specifics
        "chunk_tokens": chunk_tokens,
        "start_token": start_token,
        "end_token": end_token,
        "chunk_type": meta.get("chunk_type", "packed"),
        "overlap_with_prev": meta.get("overlap_with_prev", 0),
        "overlap_with_next": meta.get("overlap_with_next", 0),

        # Content
        "text": text,

        # Timestamp
        "chunked_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }


def chunk_edgar_document(jsonl_path: Path, config: Dict) -> List[Dict]:
    """
    Chunk EDGAR filing (10-K or 10-Q) using section-aware strategy.

    Strategy:
        1. Read JSONL atomic units (paragraphs)
        2. Group paragraphs by section_id (from existing metadata)
        3. For each section:
           a. Combine paragraph texts
           b. Split into paragraph units
           c. Pack to target_chunk_tokens
           d. If chunk <= max_chunk_tokens: keep as single chunk
           e. If chunk > max_chunk_tokens: window with overlap

    Args:
        jsonl_path: Path to parsed JSONL file
        config: Chunking configuration dict

    Returns:
        List of chunk dictionaries
    """
    logger.info(f"Chunking EDGAR filing: {jsonl_path}")

    # Get tokenizer
    enc, dec = get_tokenizer(config["tokenizer"])

    # Read JSONL atomic units
    units = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                units.append(json.loads(line))

    if not units:
        logger.warning("No units found in JSONL file")
        return []

    # Group by section_id
    sections = {}
    for unit in units:
        section_id = unit.get("section_id") or "FULL"
        section_title = unit.get("section_title") or "Full Document"

        if section_id not in sections:
            sections[section_id] = {
                "units": [],
                "title": section_title
            }

        sections[section_id]["units"].append(unit)

    logger.info(f"Grouped {len(units)} paragraphs into {len(sections)} sections")

    # Chunk each section
    chunks = []
    chunk_index = 0

    for section_id, section_data in sections.items():
        section_units = section_data["units"]
        section_title = section_data["title"]

        logger.debug(f"Chunking section: {section_id} ({len(section_units)} paragraphs)")

        # Combine all paragraph texts in section
        section_text = combine_texts(section_units)

        # Split into paragraph units
        para_texts = paragraph_units(section_text)

        # Pack to target
        packed = pack_units(
            para_texts,
            enc,
            config["target_chunk_tokens"],
            config["min_chunk_tokens"]
        )

        # Process each packed chunk
        for chunk_text in packed:
            chunk_tokens = count_tokens(chunk_text, enc)

            if chunk_tokens <= config["max_chunk_tokens"]:
                # Chunk is within limit
                chunk = build_chunk(
                    chunk_text,
                    section_units,
                    chunk_index,
                    {
                        "section_id": section_id,
                        "section_title": section_title,
                        "enc": enc,
                        "chunk_type": "section_packed"
                    }
                )
                chunks.append(chunk)
                chunk_index += 1

            else:
                # Chunk exceeds max, apply windowing with overlap
                logger.debug(f"Windowing large chunk: {chunk_tokens} tokens")

                windows = window_with_overlap(
                    chunk_text,
                    enc,
                    dec,
                    config["max_chunk_tokens"],
                    config["overlap_ratio"]
                )

                # Compute overlaps
                overlaps = compute_overlap_spans(windows)

                # Create chunk for each window
                for (start, end), (overlap_prev, overlap_next) in zip(windows, overlaps):
                    window_text = extract_window_text(chunk_text, (start, end), enc, dec)

                    chunk = build_chunk(
                        window_text,
                        section_units,
                        chunk_index,
                        {
                            "section_id": section_id,
                            "section_title": section_title,
                            "enc": enc,
                            "chunk_type": "windowed_overlap",
                            "overlap_with_prev": overlap_prev,
                            "overlap_with_next": overlap_next
                        },
                        start,
                        end
                    )
                    chunks.append(chunk)
                    chunk_index += 1

    logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
    return chunks


def chunk_transcript_document(jsonl_path: Path, config: Dict) -> List[Dict]:
    """
    Chunk earnings transcript using speaker/phase-aware strategy.

    Strategy:
        1. Read JSONL atomic units (utterances)
        2. Group by phase (prepared_remarks vs qa)
        3. For Q&A: group by exchange_id (question + answer pairs)
        4. Pack utterances to target, window if needed
        5. Preserve speaker metadata

    Args:
        jsonl_path: Path to parsed JSONL file
        config: Chunking configuration dict

    Returns:
        List of chunk dictionaries
    """
    logger.info(f"Chunking earnings transcript: {jsonl_path}")

    # Get tokenizer
    enc, dec = get_tokenizer(config["tokenizer"])

    # Read JSONL atomic units
    units = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                units.append(json.loads(line))

    if not units:
        logger.warning("No units found in JSONL file")
        return []

    # Group by phase
    phases = {"prepared_remarks": [], "qa": []}
    for unit in units:
        phase = unit.get("phase", "prepared_remarks")
        if phase in phases:
            phases[phase].append(unit)

    logger.info(
        f"Grouped {len(units)} utterances: "
        f"{len(phases['prepared_remarks'])} prepared remarks, "
        f"{len(phases['qa'])} Q&A"
    )

    chunks = []
    chunk_index = 0

    # Chunk prepared remarks
    if phases["prepared_remarks"]:
        prepared_text = combine_texts(phases["prepared_remarks"])
        para_texts = paragraph_units(prepared_text)

        packed = pack_units(
            para_texts,
            enc,
            config["target_chunk_tokens"],
            config["min_chunk_tokens"]
        )

        for chunk_text in packed:
            chunk_tokens = count_tokens(chunk_text, enc)

            if chunk_tokens <= config["max_chunk_tokens"]:
                chunk = build_chunk(
                    chunk_text,
                    phases["prepared_remarks"],
                    chunk_index,
                    {"phase": "prepared_remarks", "enc": enc, "chunk_type": "prepared_packed"}
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Window if needed
                windows = window_with_overlap(
                    chunk_text, enc, dec,
                    config["max_chunk_tokens"],
                    config["overlap_ratio"]
                )
                overlaps = compute_overlap_spans(windows)

                for (start, end), (overlap_prev, overlap_next) in zip(windows, overlaps):
                    window_text = extract_window_text(chunk_text, (start, end), enc, dec)
                    chunk = build_chunk(
                        window_text,
                        phases["prepared_remarks"],
                        chunk_index,
                        {
                            "phase": "prepared_remarks",
                            "enc": enc,
                            "chunk_type": "windowed_overlap",
                            "overlap_with_prev": overlap_prev,
                            "overlap_with_next": overlap_next
                        },
                        start,
                        end
                    )
                    chunks.append(chunk)
                    chunk_index += 1

    # Chunk Q&A section
    if phases["qa"]:
        # Group by exchange
        exchanges = group_by_exchange(phases["qa"])

        for exchange_units in exchanges:
            exchange_text = combine_texts(exchange_units)
            exchange_tokens = count_tokens(exchange_text, enc)

            # Get speaker info from first unit (question)
            first_unit = exchange_units[0] if exchange_units else {}
            speaker = first_unit.get("speaker_name", "Unknown")
            speaker_role = first_unit.get("speaker_role", "unknown")

            if exchange_tokens <= config["max_chunk_tokens"]:
                # Keep exchange together
                chunk = build_chunk(
                    exchange_text,
                    exchange_units,
                    chunk_index,
                    {
                        "phase": "qa",
                        "speaker": speaker,
                        "speaker_role": speaker_role,
                        "enc": enc,
                        "chunk_type": "qa_exchange"
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Exchange too large, window with overlap
                windows = window_with_overlap(
                    exchange_text, enc, dec,
                    config["max_chunk_tokens"],
                    config["overlap_ratio"]
                )
                overlaps = compute_overlap_spans(windows)

                for (start, end), (overlap_prev, overlap_next) in zip(windows, overlaps):
                    window_text = extract_window_text(exchange_text, (start, end), enc, dec)
                    chunk = build_chunk(
                        window_text,
                        exchange_units,
                        chunk_index,
                        {
                            "phase": "qa",
                            "speaker": speaker,
                            "speaker_role": speaker_role,
                            "enc": enc,
                            "chunk_type": "windowed_overlap",
                            "overlap_with_prev": overlap_prev,
                            "overlap_with_next": overlap_next
                        },
                        start,
                        end
                    )
                    chunks.append(chunk)
                    chunk_index += 1

    logger.info(f"Created {len(chunks)} chunks from transcript")
    return chunks


def chunk_document(jsonl_path: Path, config: Dict) -> List[Dict]:
    """
    Main entry point for document chunking.

    Routes to appropriate chunking strategy based on doc_type:
    - 10-K, 10-Q: chunk_edgar_document()
    - earnings_transcript: chunk_transcript_document()

    Args:
        jsonl_path: Path to parsed JSONL file
        config: Chunking configuration dict

    Returns:
        List of chunk dictionaries

    Example:
        >>> config = load_config('configs/chunking.yaml')
        >>> chunks = chunk_document(Path('data_parsed/AAPL/edgar/AAPL_10K_2024.jsonl'), config)
        >>> len(chunks)
        145
    """
    # Peek at first line to determine doc_type
    with open(jsonl_path, 'r') as f:
        first_line = f.readline()
        if not first_line.strip():
            logger.error("Empty JSONL file")
            return []

        first_unit = json.loads(first_line)
        doc_type = first_unit.get("doc_type", "unknown")

    # Route to appropriate strategy
    if doc_type in ["10-K", "10-Q"]:
        return chunk_edgar_document(jsonl_path, config)
    elif doc_type == "earnings_transcript":
        return chunk_transcript_document(jsonl_path, config)
    else:
        logger.error(f"Unknown doc_type: {doc_type}")
        return []


def save_chunked_output(chunks: List[Dict], output_path: Path):
    """
    Save chunks to JSONL file.

    Args:
        chunks: List of chunk dictionaries
        output_path: Path to output JSONL file

    Example:
        >>> chunks = chunk_document(input_path, config)
        >>> save_chunked_output(chunks, Path('data_chunked/AAPL/edgar/AAPL_10K_2024_chunks_600.jsonl'))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for chunk in chunks:
            # Remove encoder from metadata before saving
            if 'enc' in chunk:
                del chunk['enc']

            f.write(json.dumps(chunk) + '\n')

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")
