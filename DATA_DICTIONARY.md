# Data Dictionary - FinSearch AI

**Version**: 2.0
**Last Updated**: 2025-01-09

This document defines the unified metadata schema for all document types in FinSearch AI.

---

## Overview

All parsed documents (EDGAR filings and earnings transcripts) are stored in JSONL format with standardized metadata. Each line in a JSONL file represents one **chunk** (paragraph or utterance) with complete metadata.

### File Structure

```
data_parsed/
├── {TICKER}/
│   ├── edgar/
│   │   ├── {DOC_ID}.jsonl     # Paragraph units
│   │   ├── {DOC_ID}.md        # Human-readable export
│   │   └── tables/
│   │       └── {DOC_ID}_table_N.csv
│   └── earnings/
│       ├── {YEAR}_Q{N}.jsonl  # Utterance units
│       └── {YEAR}_Q{N}.md     # Human-readable export
```

---

## Core Field Definitions

### Identifiers

| Field | Type | Required | Description | Examples |
|-------|------|----------|-------------|----------|
| `doc_id` | string | ✓ | Unique document identifier | `AAPL_10K_2024`<br>`AAPL_TRANSCRIPT_2025_Q4` |
| `chunk_id` | string | ✓ | Unique chunk identifier | `AAPL_10K_2024_0001`<br>`AAPL_TRANSCRIPT_2025_Q4_u_0042` |

**Format Rules**:
- `doc_id`: `{TICKER}_{DOCTYPE}_{YEAR}[_{QUARTER}]`
- `chunk_id`: `{DOC_ID}_{UNIT_INDEX}` or `{DOC_ID}_u_{UNIT_INDEX}` for utterances

---

### Company Information

| Field | Type | Required | Description | Examples |
|-------|------|----------|-------------|----------|
| `ticker` | string | ✓ | Stock ticker symbol | `AAPL`, `MSFT`, `GOOGL` |
| `company` | string | ✓ | Company identifier (currently same as ticker) | `AAPL`, `MSFT`, `GOOGL` |

**Note**: Currently `company` is set to `ticker`. Can be expanded to full company names in future versions.

---

### Document Type & Period

| Field | Type | Required | Description | Examples | Enum Values |
|-------|------|----------|-------------|----------|-------------|
| `doc_type` | string | ✓ | Document type | `10-K`, `10-Q`, `earnings_transcript` | `10-K`, `10-Q`, `earnings_transcript` |
| `fiscal_year` | integer | ✓ | Fiscal year | `2024`, `2025` | — |
| `quarter` | string | ✓* | Fiscal quarter | `FY`, `Q1`, `Q2`, `Q3`, `Q4` | `FY`, `Q1`, `Q2`, `Q3`, `Q4` |
| `period` | string | ✓ | Standardized period identifier | `2024-FY`, `2025-Q4` | — |
| `filing_date` | string | ✓ | Filing/transcript date | `2024-10-31`, `2025-08-01` | `YYYY-MM-DD` |

**Format Rules**:
- `period`: Computed as `{fiscal_year}-{quarter}`
- `quarter`: `FY` for 10-K annual reports, `Q1`-`Q4` for 10-Q and transcripts
- `filing_date`: ISO 8601 date format (`YYYY-MM-DD`)

*Required but may be `null` for some 10-K filings without quarterly data

---

### Section & Structure

| Field | Type | Required | Description | Examples | Applicable To |
|-------|------|----------|-------------|----------|---------------|
| `section_id` | string | — | Section identifier | `Item 1A`, `Item 7`, `Part I` | EDGAR filings |
| `section_title` | string | — | Full section title | `Risk Factors`, `Management's Discussion and Analysis` | EDGAR filings |

**Note**: These fields are `null` for earnings transcripts

**Common Section IDs**:

**10-K Filings**:
- `Item 1`: Business
- `Item 1A`: Risk Factors
- `Item 7`: Management's Discussion and Analysis (MD&A)
- `Item 8`: Financial Statements
- `Item 10`: Directors, Executive Officers, and Corporate Governance
- `Item 11`: Executive Compensation
- ... (see SECTIONS_10K in code for complete list)

**10-Q Filings**:
- `Part I`: Financial Information
- `Item 1`: Financial Statements
- `Item 2`: Management's Discussion and Analysis
- `Part II`: Other Information
- ... (see SECTIONS_10Q in code for complete list)

---

### Chunk Information

| Field | Type | Required | Description | Examples | Enum Values |
|-------|------|----------|-------------|----------|-------------|
| `unit_type` | string | ✓ | Type of text unit | `paragraph`, `utterance` | `paragraph`, `utterance` |
| `unit_index` | integer | ✓ | Sequential index within document (0-based) | `0`, `42`, `830` | — |
| `text` | string | ✓ | Actual text content | *Full paragraph or utterance text* | — |
| `char_count` | integer | ✓ | Character count | `1234`, `5678` | — |
| `word_count` | integer | ✓ | Word count | `180`, `450` | — |
| `token_count` | integer | — | Token count (tiktoken cl100k_base) | `234`, `567` | — |

**Note**:
- `unit_type`: `paragraph` for EDGAR filings, `utterance` for transcripts
- `token_count`: Only available for transcripts (uses tiktoken)

---

### Source Tracking

| Field | Type | Required | Description | Examples |
|-------|------|----------|-------------|----------|
| `source_file` | string | ✓ | Path to original source file | `data/edgar/AAPL/raw_10-K_2024.htm`<br>`data/earnings_calls_manual/AAPL/AAPL_FY2025_Q4.pdf` |
| `parsed_at` | string | ✓ | Timestamp when parsed (ISO 8601) | `2025-01-09T14:30:00Z` |

---

### Transcript-Specific Fields

These fields are **only populated for earnings transcripts** (doc_type = `earnings_transcript`).
For EDGAR filings, these fields are `null`.

| Field | Type | Required | Description | Examples | Enum Values |
|-------|------|----------|-------------|----------|-------------|
| `phase` | string | — | Call phase | `prepared_remarks`, `qa` | `prepared_remarks`, `qa` |
| `speaker_name` | string | — | Speaker name | `Timothy Cook`, `Luca Maestri` | — |
| `speaker_role` | string | — | Speaker role | `CEO`, `CFO`, `analyst` | See **Speaker Roles** below |
| `speaker_firm` | string | — | Analyst firm (for analysts only) | `Goldman Sachs`, `Morgan Stanley` | — |
| `utterance_id` | string | — | Utterance identifier | `u_0000`, `u_0042` | `u_NNNN` |
| `utterance_type` | string | — | Type of utterance | `statement`, `question`, `answer` | `statement`, `question`, `answer` |
| `exchange_id` | string | — | Q&A exchange ID | `ex_001`, `ex_013` | `ex_NNN` |
| `exchange_role` | string | — | Role in Q&A exchange | `question`, `answer` | `question`, `answer` |

**Speaker Roles**:
- `CEO`: Chief Executive Officer
- `CFO`: Chief Financial Officer
- `COO`: Chief Operating Officer
- `CTO`: Chief Technology Officer
- `executive`: Other executive roles
- `analyst`: Financial analyst
- `operator`: Call operator/moderator
- `unknown`: Role not determined

**Exchange Pairing**:
- Questions and answers in Q&A section are paired using `exchange_id`
- Each exchange has one `question` and one or more `answer` utterances
- `exchange_role` indicates position within the exchange

---

## Field Requirements by Document Type

### EDGAR Filings (10-K, 10-Q)

**Required Fields**:
```json
{
  "doc_id": "string",
  "chunk_id": "string",
  "ticker": "string",
  "company": "string",
  "doc_type": "10-K" | "10-Q",
  "fiscal_year": "integer",
  "quarter": "string",
  "period": "string",
  "filing_date": "string (YYYY-MM-DD)",
  "section_id": "string | null",
  "section_title": "string | null",
  "unit_type": "paragraph",
  "unit_index": "integer",
  "text": "string",
  "char_count": "integer",
  "word_count": "integer",
  "source_file": "string",
  "parsed_at": "string (ISO 8601)"
}
```

**Transcript-specific fields**: All `null`

### Earnings Transcripts

**Required Fields**:
```json
{
  "doc_id": "string",
  "chunk_id": "string",
  "ticker": "string",
  "company": "string",
  "doc_type": "earnings_transcript",
  "fiscal_year": "integer",
  "quarter": "string",
  "period": "string",
  "filing_date": "string (YYYY-MM-DD)",
  "unit_type": "utterance",
  "unit_index": "integer",
  "text": "string",
  "char_count": "integer",
  "word_count": "integer",
  "token_count": "integer",
  "source_file": "string",
  "parsed_at": "string (ISO 8601)",
  "phase": "string",
  "speaker_name": "string",
  "speaker_role": "string",
  "speaker_firm": "string | null",
  "utterance_id": "string",
  "utterance_type": "string",
  "exchange_id": "string | null",
  "exchange_role": "string | null"
}
```

**EDGAR-specific fields**: `section_id` and `section_title` are `null`

---

## Example Records

### EDGAR Filing Chunk (10-K)

```json
{
  "doc_id": "AAPL_10K_2024",
  "chunk_id": "AAPL_10K_2024_0123",
  "ticker": "AAPL",
  "company": "AAPL",
  "doc_type": "10-K",
  "fiscal_year": 2024,
  "quarter": "FY",
  "period": "2024-FY",
  "filing_date": "2024-10-31",
  "section_id": "Item 1A",
  "section_title": "Risk Factors",
  "unit_type": "paragraph",
  "unit_index": 123,
  "text": "The Company's business, reputation, results of operations...",
  "char_count": 1234,
  "word_count": 180,
  "source_file": "data/edgar/AAPL/raw_10-K_2024.htm",
  "parsed_at": "2025-01-09T14:30:00Z",
  "phase": null,
  "speaker_name": null,
  "speaker_role": null,
  "speaker_firm": null,
  "utterance_id": null,
  "utterance_type": null,
  "token_count": null,
  "exchange_id": null,
  "exchange_role": null
}
```

### Earnings Transcript Chunk (Q&A Question)

```json
{
  "doc_id": "AAPL_TRANSCRIPT_2025_Q4",
  "chunk_id": "AAPL_TRANSCRIPT_2025_Q4_u_0042",
  "ticker": "AAPL",
  "company": "AAPL",
  "doc_type": "earnings_transcript",
  "fiscal_year": 2025,
  "quarter": "Q4",
  "period": "2025-Q4",
  "filing_date": "2025-12-01",
  "section_id": null,
  "section_title": null,
  "unit_type": "utterance",
  "unit_index": 42,
  "text": "Thanks for taking my question. Could you provide more color on the iPhone sales trends in China?",
  "char_count": 96,
  "word_count": 17,
  "token_count": 23,
  "source_file": "data/earnings_calls_manual/AAPL/AAPL_FY2025_Q4.pdf",
  "parsed_at": "2025-01-09T14:30:00Z",
  "phase": "qa",
  "speaker_name": "Shannon Cross",
  "speaker_role": "analyst",
  "speaker_firm": "Goldman Sachs",
  "utterance_id": "u_0042",
  "utterance_type": "question",
  "exchange_id": "ex_013",
  "exchange_role": "question"
}
```

---

## Validation Rules

### Enum Constraints

**doc_type**:
- `10-K`: Annual report
- `10-Q`: Quarterly report
- `earnings_transcript`: Earnings call transcript

**quarter**:
- `FY`: Full year (10-K only)
- `Q1`, `Q2`, `Q3`, `Q4`: Fiscal quarters

**phase** (transcripts only):
- `prepared_remarks`: Prepared presentation
- `qa`: Question and answer session

**speaker_role** (transcripts only):
- `CEO`, `CFO`, `COO`, `CTO`, `executive`, `analyst`, `operator`, `unknown`

**utterance_type** (transcripts only):
- `statement`: General statement
- `question`: Analyst question
- `answer`: Executive answer

**unit_type**:
- `paragraph`: Text paragraph (EDGAR)
- `utterance`: Speech utterance (transcript)

### Data Integrity Rules

1. **doc_id Format**: Must follow pattern `{TICKER}_{DOCTYPE}_{YEAR}[_{QUARTER}]`
2. **chunk_id Uniqueness**: Must be unique within dataset
3. **period Computation**: Must equal `{fiscal_year}-{quarter}`
4. **Date Format**: `filing_date` and `parsed_at` must be valid ISO 8601
5. **Null Consistency**:
   - EDGAR filings: transcript-specific fields must be `null`
   - Transcripts: `section_id` and `section_title` must be `null`

---

## Schema Version History

### Version 2.0 (2025-01-09)
- **Added**: `chunk_id`, `company`, `fiscal_year`, `quarter`, `period`, `source_file`, `parsed_at`
- **Changed**: Unified schema across EDGAR and transcript documents
- **Changed**: Standardized quarter format to string (`Q1`-`Q4`, `FY`)
- **Changed**: `doc_type` for transcripts to `earnings_transcript`
- **Removed**: Inconsistent field formats between document types

### Version 1.0 (2025-01-08)
- Initial JSONL format with basic metadata
- Separate schemas for EDGAR and transcripts

---

## Implementation Reference

See `backend/app/services/data_ingestion/metadata_schema.py` for:
- Enum definitions
- Validation functions
- Schema helpers (compute_period, compute_chunk_id, get_current_timestamp)

---

## Notes for RAG/Vector Database Integration

### Recommended Metadata Filters

**By Company**:
```python
filter = {"ticker": "AAPL"}
```

**By Document Type**:
```python
filter = {"doc_type": "10-K"}  # Annual reports only
filter = {"doc_type": {"$in": ["10-K", "10-Q"]}}  # All EDGAR filings
```

**By Time Period**:
```python
filter = {"fiscal_year": 2024}
filter = {"period": "2024-Q3"}
filter = {"fiscal_year": {"$gte": 2023}}  # 2023 onwards
```

**By Section** (EDGAR only):
```python
filter = {"section_id": "Item 1A"}  # Risk factors
filter = {"section_id": "Item 7"}  # MD&A
```

**By Speaker** (Transcripts only):
```python
filter = {"speaker_role": "CEO"}
filter = {"phase": "qa"}  # Q&A section only
filter = {"speaker_firm": "Goldman Sachs"}
```

**Combined Filters**:
```python
filter = {
    "ticker": "AAPL",
    "fiscal_year": 2024,
    "doc_type": "10-K",
    "section_id": "Item 7"
}
```

---

## Contact

For questions about this schema or to report issues:
- GitHub: [FinSearch AI Issues](https://github.com/syang620/FinSearch-AI/issues)
