"""
Section and Phase Splitting

Utilities for splitting documents into semantic sections:
- EDGAR filings: split by Item headers (Item 1A, Item 7, etc.)
- Earnings transcripts: split by phase (Prepared Remarks vs Q&A)
"""

import re
from typing import List, Dict, Callable
import logging

logger = logging.getLogger(__name__)


def split_filing_sections(text: str, regex: str, enc: Callable[[str], list]) -> List[Dict]:
    """
    Split EDGAR filing text into sections based on Item headers.

    Args:
        text: Full document text (markdown or plain text)
        regex: Section header pattern (e.g., r'(?mi)^\s*Item\s+(\d+[A]?)\.\s+(.+?)\s*$')
            Group 1: Item number (e.g., "1A")
            Group 2: Section title (e.g., "Risk Factors")
        enc: Token encoder function (for logging/debugging)

    Returns:
        List of section dictionaries with:
            - section_id: str (e.g., "Item 1A")
            - title: str (e.g., "Risk Factors")
            - text: str (section content)

    Example:
        >>> text = "Item 1A. Risk Factors\\n\\nOur business faces risks...\\n\\nItem 7. MD&A\\n\\nRevenue increased..."
        >>> sections = split_filing_sections(text, r'(?mi)^\s*Item\s+(\d+[A]?)\.\s+(.+?)\s*$', enc)
        >>> len(sections)
        2
        >>> sections[0]['section_id']
        'Item 1A'
        >>> sections[0]['title']
        'Risk Factors'
    """
    try:
        matches = list(re.finditer(regex, text, re.MULTILINE | re.IGNORECASE))
    except re.error as e:
        logger.error(f"Invalid regex pattern: {regex}. Error: {e}")
        matches = []

    if not matches:
        logger.warning("No section headers found. Returning full document as single section.")
        return [{
            "section_id": "FULL",
            "title": "Full Document",
            "text": text
        }]

    blocks = []
    for i, match in enumerate(matches):
        # Section content starts after the header line
        start = match.end()

        # Section ends at next header (or end of document)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        # Extract section ID and title from regex groups
        section_id = match.group(1)  # e.g., "1A", "7", "15"
        title = match.group(2).strip()  # e.g., "Risk Factors"

        section_text = text[start:end].strip()

        blocks.append({
            "section_id": f"Item {section_id}",
            "title": title,
            "text": section_text
        })

        logger.debug(
            f"Extracted section: Item {section_id} - {title} "
            f"({len(section_text)} chars)"
        )

    logger.info(f"Split filing into {len(blocks)} sections")
    return blocks


def split_transcript(text: str, phase_headers: List[str]) -> List[Dict]:
    """
    Split earnings transcript into phases (Prepared Remarks vs Q&A).

    Args:
        text: Full transcript text
        phase_headers: List of regex patterns for phase detection, e.g.:
            - r'(?i)prepared remarks'
            - r'(?i)question\s*&\s*answer'

    Returns:
        List of phase dictionaries with:
            - phase: str ("Prepared Remarks" or "Q&A")
            - text: str (phase content)

    Note:
        If no phase headers are detected, returns full transcript as single phase.

    Example:
        >>> text = "Prepared Remarks\\n\\nCEO: Great quarter...\\n\\nQuestions & Answers\\n\\nAnalyst: ..."
        >>> phases = split_transcript(text, [r'(?i)prepared remarks', r'(?i)question.+answer'])
        >>> len(phases)
        2
        >>> phases[0]['phase']
        'Prepared Remarks'
    """
    # Try to find phase boundaries
    phase_positions = []

    for pattern in phase_headers:
        try:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                # Determine phase type from match
                match_text = match.group(0).lower()

                if 'prepared' in match_text or 'remark' in match_text:
                    phase_type = "Prepared Remarks"
                elif 'question' in match_text or 'q&a' in match_text or 'q & a' in match_text:
                    phase_type = "Q&A"
                else:
                    phase_type = "Unknown"

                phase_positions.append((match.end(), phase_type))

        except re.error as e:
            logger.warning(f"Invalid phase header pattern: {pattern}. Error: {e}")
            continue

    if not phase_positions:
        logger.warning("No phase headers detected. Returning full transcript as single phase.")
        return [{
            "phase": "Full Transcript",
            "text": text
        }]

    # Sort by position
    phase_positions.sort(key=lambda x: x[0])

    # Extract phase blocks
    blocks = []
    for i, (start_pos, phase_type) in enumerate(phase_positions):
        # Phase ends at next phase (or end of document)
        end_pos = phase_positions[i + 1][0] if i + 1 < len(phase_positions) else len(text)

        phase_text = text[start_pos:end_pos].strip()

        blocks.append({
            "phase": phase_type,
            "text": phase_text
        })

        logger.debug(f"Extracted phase: {phase_type} ({len(phase_text)} chars)")

    logger.info(f"Split transcript into {len(blocks)} phases")
    return blocks


def group_by_speaker(utterances: List[Dict]) -> List[List[Dict]]:
    """
    Group transcript utterances by speaker for packing.

    Args:
        utterances: List of utterance dicts from JSONL with 'speaker_name' field

    Returns:
        List of speaker groups (each group is a list of consecutive utterances
        from the same speaker)

    Example:
        >>> utterances = [
        ...     {"speaker_name": "CEO", "text": "Hello"},
        ...     {"speaker_name": "CEO", "text": "Welcome"},
        ...     {"speaker_name": "CFO", "text": "Thank you"},
        ... ]
        >>> groups = group_by_speaker(utterances)
        >>> len(groups)
        2
        >>> len(groups[0])  # CEO group
        2
    """
    if not utterances:
        return []

    groups = []
    current_group = [utterances[0]]
    current_speaker = utterances[0].get('speaker_name', 'Unknown')

    for utt in utterances[1:]:
        speaker = utt.get('speaker_name', 'Unknown')

        if speaker == current_speaker:
            # Same speaker, add to current group
            current_group.append(utt)
        else:
            # New speaker, start new group
            groups.append(current_group)
            current_group = [utt]
            current_speaker = speaker

    # Don't forget the last group
    if current_group:
        groups.append(current_group)

    logger.debug(f"Grouped {len(utterances)} utterances into {len(groups)} speaker groups")
    return groups


def group_by_exchange(utterances: List[Dict]) -> List[List[Dict]]:
    """
    Group Q&A utterances by exchange_id (question + answer pairs).

    Args:
        utterances: List of Q&A utterance dicts with 'exchange_id' field

    Returns:
        List of exchange groups (each group is all utterances for one Q&A exchange)

    Example:
        >>> utterances = [
        ...     {"exchange_id": "ex_001", "exchange_role": "question", "text": "How was revenue?"},
        ...     {"exchange_id": "ex_001", "exchange_role": "answer", "text": "Great, up 20%"},
        ...     {"exchange_id": "ex_002", "exchange_role": "question", "text": "What about margins?"},
        ... ]
        >>> exchanges = group_by_exchange(utterances)
        >>> len(exchanges)
        2
        >>> len(exchanges[0])  # First exchange (Q + A)
        2
    """
    if not utterances:
        return []

    # Group by exchange_id
    exchange_dict = {}
    for utt in utterances:
        exchange_id = utt.get('exchange_id')

        if not exchange_id:
            # Utterance not part of an exchange (e.g., operator)
            continue

        if exchange_id not in exchange_dict:
            exchange_dict[exchange_id] = []

        exchange_dict[exchange_id].append(utt)

    # Convert to list (maintain order)
    exchanges = list(exchange_dict.values())

    logger.debug(f"Grouped {len(utterances)} Q&A utterances into {len(exchanges)} exchanges")
    return exchanges
