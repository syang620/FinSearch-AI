"""
Unified Metadata Schema for FinSearch AI

Defines the standard metadata schema for all document types (EDGAR filings and earnings transcripts).
All chunks and documents must conform to this schema for consistent RAG retrieval and filtering.
"""

from enum import Enum
from typing import Optional, Literal, List, Tuple
from datetime import datetime


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class DocType(str, Enum):
    """Document type enumeration"""
    FILING_10K = "10-K"
    FILING_10Q = "10-Q"
    TRANSCRIPT = "earnings_transcript"


class Quarter(str, Enum):
    """Fiscal quarter enumeration"""
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    FY = "FY"  # Full year (for 10-K)


class TranscriptPhase(str, Enum):
    """Earnings call transcript phases"""
    PREPARED_REMARKS = "prepared_remarks"
    QA = "qa"


class SpeakerRole(str, Enum):
    """Speaker roles in earnings transcripts"""
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    CTO = "CTO"
    EXECUTIVE = "executive"
    ANALYST = "analyst"
    OPERATOR = "operator"
    UNKNOWN = "unknown"


class UnitType(str, Enum):
    """Unit type for document chunks"""
    PARAGRAPH = "paragraph"
    UTTERANCE = "utterance"


# ============================================================================
# UNIFIED METADATA SCHEMA
# ============================================================================

UNIFIED_METADATA_SCHEMA = {
    # ========================================================================
    # CORE IDENTIFIERS (Required for all documents)
    # ========================================================================
    "doc_id": {
        "type": "string",
        "required": True,
        "description": "Unique document identifier",
        "format": "TICKER_DOCTYPE_YEAR[_QUARTER]",
        "examples": ["AAPL_10K_2024", "AAPL_10Q_2024_Q3", "AAPL_TRANSCRIPT_2025_Q4"]
    },

    "chunk_id": {
        "type": "string",
        "required": True,
        "description": "Unique chunk identifier within document",
        "format": "DOC_ID_UNIT_INDEX",
        "examples": ["AAPL_10K_2024_0001", "AAPL_TRANSCRIPT_2025_Q4_u_0042"]
    },

    # ========================================================================
    # COMPANY INFORMATION (Required for all documents)
    # ========================================================================
    "ticker": {
        "type": "string",
        "required": True,
        "description": "Stock ticker symbol (also used as company identifier)",
        "examples": ["AAPL", "MSFT", "GOOGL"]
    },

    "company": {
        "type": "string",
        "required": True,
        "description": "Company name (same as ticker for now)",
        "examples": ["AAPL", "MSFT", "GOOGL"],
        "note": "Currently set to ticker; can be expanded to full name later"
    },

    # ========================================================================
    # DOCUMENT TYPE AND PERIOD (Required for all documents)
    # ========================================================================
    "doc_type": {
        "type": "string",
        "required": True,
        "enum": ["10-K", "10-Q", "earnings_transcript"],
        "description": "Document type classification"
    },

    "fiscal_year": {
        "type": "integer",
        "required": True,
        "description": "Fiscal year of the document",
        "examples": [2024, 2025]
    },

    "quarter": {
        "type": "string",
        "required": False,
        "enum": ["Q1", "Q2", "Q3", "Q4", "FY"],
        "description": "Fiscal quarter (FY for 10-K annual reports)",
        "note": "10-K uses 'FY', 10-Q and transcripts use Q1-Q4"
    },

    "period": {
        "type": "string",
        "required": True,
        "description": "Standardized period identifier",
        "format": "YYYY-QN or YYYY-FY",
        "examples": ["2024-FY", "2024-Q3", "2025-Q4"],
        "note": "Computed from fiscal_year + quarter"
    },

    "filing_date": {
        "type": "string",
        "required": True,
        "description": "Date of filing or transcript",
        "format": "YYYY-MM-DD",
        "examples": ["2024-10-31", "2025-08-01"]
    },

    # ========================================================================
    # SECTION/STRUCTURE (Required for all documents)
    # ========================================================================
    "section_id": {
        "type": "string",
        "required": False,
        "description": "Section identifier (Item 1A, Item 7, etc. for filings)",
        "examples": ["Item 1A", "Item 7", "Part I"],
        "note": "null for transcript utterances"
    },

    "section_title": {
        "type": "string",
        "required": False,
        "description": "Full section title",
        "examples": ["Risk Factors", "Management's Discussion and Analysis"],
        "note": "null for transcript utterances"
    },

    # ========================================================================
    # CHUNK INFORMATION (Required for all documents)
    # ========================================================================
    "unit_type": {
        "type": "string",
        "required": True,
        "enum": ["paragraph", "utterance"],
        "description": "Type of text unit (paragraph for filings, utterance for transcripts)"
    },

    "unit_index": {
        "type": "integer",
        "required": True,
        "description": "Sequential index of unit within document (0-based)",
        "examples": [0, 1, 2, 42]
    },

    "text": {
        "type": "string",
        "required": True,
        "description": "The actual text content of the chunk"
    },

    "char_count": {
        "type": "integer",
        "required": True,
        "description": "Character count of the text"
    },

    "word_count": {
        "type": "integer",
        "required": True,
        "description": "Word count of the text"
    },

    "token_count": {
        "type": "integer",
        "required": False,
        "description": "Token count using tiktoken (for LLM context management)",
        "note": "Only available for transcripts currently"
    },

    # ========================================================================
    # SOURCE TRACKING (Required for all documents)
    # ========================================================================
    "source_file": {
        "type": "string",
        "required": True,
        "description": "Original source file path",
        "examples": ["data/edgar/AAPL/raw_10-K_2024.htm", "data/earnings_calls_manual/AAPL/AAPL_FY2025_Q4.pdf"]
    },

    "parsed_at": {
        "type": "string",
        "required": True,
        "description": "Timestamp when document was parsed",
        "format": "ISO 8601 datetime",
        "examples": ["2025-01-15T14:30:00Z"]
    },

    # ========================================================================
    # TRANSCRIPT-SPECIFIC FIELDS (Only for earnings_transcript)
    # ========================================================================
    "phase": {
        "type": "string",
        "required": False,
        "enum": ["prepared_remarks", "qa"],
        "description": "Phase of earnings call (for transcripts only)",
        "note": "null for EDGAR filings"
    },

    "speaker_name": {
        "type": "string",
        "required": False,
        "description": "Name of speaker (for transcripts only)",
        "examples": ["Timothy Cook", "Luca Maestri"],
        "note": "null for EDGAR filings"
    },

    "speaker_role": {
        "type": "string",
        "required": False,
        "enum": ["CEO", "CFO", "COO", "CTO", "executive", "analyst", "operator", "unknown"],
        "description": "Role of speaker (for transcripts only)",
        "note": "null for EDGAR filings"
    },

    "speaker_firm": {
        "type": "string",
        "required": False,
        "description": "Firm name for analysts (for transcripts only)",
        "examples": ["Goldman Sachs", "Morgan Stanley"],
        "note": "null for executives and EDGAR filings"
    },

    "utterance_id": {
        "type": "string",
        "required": False,
        "description": "Unique utterance identifier within transcript",
        "format": "u_NNNN",
        "examples": ["u_0000", "u_0042"],
        "note": "null for EDGAR filings"
    },

    "utterance_type": {
        "type": "string",
        "required": False,
        "enum": ["statement", "question", "answer"],
        "description": "Type of utterance (for transcripts only)",
        "note": "null for EDGAR filings"
    },

    "exchange_id": {
        "type": "string",
        "required": False,
        "description": "Q&A exchange identifier (groups question + answer)",
        "format": "ex_NNN",
        "examples": ["ex_001", "ex_013"],
        "note": "null for prepared remarks and EDGAR filings"
    },

    "exchange_role": {
        "type": "string",
        "required": False,
        "enum": ["question", "answer"],
        "description": "Role within Q&A exchange (for transcripts only)",
        "note": "null for prepared remarks and EDGAR filings"
    }
}


# ============================================================================
# SCHEMA VALIDATION HELPERS
# ============================================================================

REQUIRED_FIELDS_ALL = [
    "doc_id",
    "chunk_id",
    "ticker",
    "company",
    "doc_type",
    "fiscal_year",
    "period",
    "filing_date",
    "unit_type",
    "unit_index",
    "text",
    "char_count",
    "word_count",
    "source_file",
    "parsed_at"
]

REQUIRED_FIELDS_EDGAR = REQUIRED_FIELDS_ALL + [
    "section_id",  # Can be null but field must exist
    "section_title"  # Can be null but field must exist
]

REQUIRED_FIELDS_TRANSCRIPT = REQUIRED_FIELDS_ALL + [
    "phase",
    "speaker_name",
    "speaker_role",
    "utterance_id",
    "utterance_type",
    "token_count"
]

# Allowed values for enums
DOC_TYPES = ["10-K", "10-Q", "earnings_transcript"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4", "FY"]
PHASES = ["prepared_remarks", "qa"]
SPEAKER_ROLES = ["CEO", "CFO", "COO", "CTO", "executive", "analyst", "operator", "unknown"]
UTTERANCE_TYPES = ["statement", "question", "answer"]
UNIT_TYPES = ["paragraph", "utterance"]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_period(fiscal_year: int, quarter: Optional[str]) -> str:
    """
    Compute standardized period string from fiscal year and quarter

    Args:
        fiscal_year: Fiscal year (e.g., 2024)
        quarter: Quarter string (Q1-Q4 or FY)

    Returns:
        Standardized period string (e.g., "2024-Q3" or "2024-FY")
    """
    if quarter:
        return f"{fiscal_year}-{quarter}"
    else:
        return f"{fiscal_year}-FY"


def compute_chunk_id(doc_id: str, unit_index: int, unit_type: str = "paragraph") -> str:
    """
    Compute unique chunk ID

    Args:
        doc_id: Document ID
        unit_index: Unit index
        unit_type: Unit type (paragraph or utterance)

    Returns:
        Chunk ID string
    """
    if unit_type == "utterance":
        return f"{doc_id}_u_{unit_index:04d}"
    else:
        return f"{doc_id}_{unit_index:04d}"


def get_current_timestamp() -> str:
    """Get current timestamp in ISO 8601 format"""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_metadata(metadata: dict, doc_type: str) -> Tuple[bool, List[str]]:
    """
    Validate metadata against schema

    Args:
        metadata: Metadata dictionary to validate
        doc_type: Document type (10-K, 10-Q, earnings_transcript)

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required fields based on doc type
    if doc_type in ["10-K", "10-Q"]:
        required = REQUIRED_FIELDS_EDGAR
    elif doc_type == "earnings_transcript":
        required = REQUIRED_FIELDS_TRANSCRIPT
    else:
        errors.append(f"Invalid doc_type: {doc_type}")
        return False, errors

    # Check for missing required fields
    for field in required:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")

    # Validate enum fields
    if "doc_type" in metadata and metadata["doc_type"] not in DOC_TYPES:
        errors.append(f"Invalid doc_type: {metadata['doc_type']}")

    if "quarter" in metadata and metadata["quarter"] and metadata["quarter"] not in QUARTERS:
        errors.append(f"Invalid quarter: {metadata['quarter']}")

    if "phase" in metadata and metadata["phase"] and metadata["phase"] not in PHASES:
        errors.append(f"Invalid phase: {metadata['phase']}")

    if "speaker_role" in metadata and metadata["speaker_role"] and metadata["speaker_role"] not in SPEAKER_ROLES:
        errors.append(f"Invalid speaker_role: {metadata['speaker_role']}")

    if "unit_type" in metadata and metadata["unit_type"] not in UNIT_TYPES:
        errors.append(f"Invalid unit_type: {metadata['unit_type']}")

    return len(errors) == 0, errors
