# SEC EDGAR Data Ingestion Guide

This guide explains how to use the SEC EDGAR data ingestion system to populate FinSearch AI with real 10-K and 10-Q filings from S&P 500 companies.

## Overview

The EDGAR ingestion system provides:
- Automated retrieval of 10-K and 10-Q filings from SEC EDGAR
- S&P 500 company list management
- Intelligent parsing of SEC filings
- Integration with the existing RAG system
- Rate-limited API access (compliant with SEC's 10 requests/second limit)

## Architecture

```
┌─────────────────┐
│  S&P 500 List   │ ──→ Ticker to CIK mapping
└─────────────────┘
         │
         ↓
┌─────────────────┐
│  EDGAR Client   │ ──→ Fetch filings from SEC API
└─────────────────┘
         │
         ↓
┌─────────────────┐
│ Filing Parser   │ ──→ Extract sections & clean text
└─────────────────┘
         │
         ↓
┌─────────────────┐
│  RAG System     │ ──→ Chunk & embed in vector store
└─────────────────┘
```

## API Endpoints

### 1. Get S&P 500 Companies

```bash
GET /api/v1/data/companies
```

**Response:**
```json
{
  "success": true,
  "count": 503,
  "companies": [
    {
      "ticker": "AAPL",
      "company_name": "Apple Inc.",
      "sector": "Information Technology",
      "industry": "Technology Hardware, Storage & Peripherals",
      "cik": "0000320193"
    },
    ...
  ]
}
```

### 2. Get Companies by Sector

```bash
GET /api/v1/data/companies/sector/{sector}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/data/companies/sector/Information%20Technology
```

### 3. Get CIK for Ticker

```bash
GET /api/v1/data/edgar/cik/{ticker}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/data/edgar/cik/AAPL
```

**Response:**
```json
{
  "ticker": "AAPL",
  "cik": "0000320193",
  "company_info": { ... }
}
```

### 4. Get Company Filings

```bash
GET /api/v1/data/edgar/filings/{ticker}?form_types=10-K,10-Q&limit=10
```

**Example:**
```bash
curl "http://localhost:8000/api/v1/data/edgar/filings/AAPL?form_types=10-K,10-Q&limit=5"
```

**Response:**
```json
{
  "ticker": "AAPL",
  "cik": "0000320193",
  "count": 5,
  "filings": [
    {
      "accession_number": "0000320193-23-000077",
      "filing_date": "2023-08-04",
      "report_date": "2023-07-01",
      "form_type": "10-Q",
      "file_number": "001-36743",
      "primary_document": "aapl-20230701.htm",
      "primary_doc_description": "10-Q"
    },
    ...
  ]
}
```

### 5. Ingest EDGAR Data

```bash
POST /api/v1/data/edgar/ingest
```

**Request Body:**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "num_filings": 3,
  "form_types": ["10-K", "10-Q"]
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/data/edgar/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT"],
    "num_filings": 2
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "Ingested 4 filings for 2 companies",
  "companies_processed": 2,
  "total_filings": 4,
  "total_chunks": 387,
  "details": {
    "results": [
      {
        "success": true,
        "ticker": "AAPL",
        "cik": "0000320193",
        "filings_processed": 2,
        "chunks_created": 195,
        "filings": [
          {
            "form_type": "10-K",
            "filing_date": "2023-11-03",
            "chunks": 98
          },
          {
            "form_type": "10-Q",
            "filing_date": "2023-08-04",
            "chunks": 97
          }
        ]
      },
      ...
    ]
  }
}
```

### 6. Ingest Sample Data

```bash
POST /api/v1/data/edgar/ingest/sample?sample_size=5&num_filings=2
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/data/edgar/ingest/sample?sample_size=5&num_filings=2"
```

This endpoint is useful for quickly testing the system with a small dataset.

### 7. Check Ingestion Status

```bash
GET /api/v1/data/edgar/status
```

## Usage Examples

### Example 1: Ingest Data for Specific Companies

```python
import requests

# Ingest filings for tech companies
response = requests.post(
    'http://localhost:8000/api/v1/data/edgar/ingest',
    json={
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        'num_filings': 3
    }
)

result = response.json()
print(f"Processed {result['companies_processed']} companies")
print(f"Created {result['total_chunks']} chunks")
```

### Example 2: Ingest All Companies in a Sector

```python
import requests

# Get all tech companies
response = requests.get(
    'http://localhost:8000/api/v1/data/companies/sector/Information Technology'
)
companies = response.json()['companies']

# Extract tickers
tickers = [c['ticker'] for c in companies[:10]]  # First 10

# Ingest
response = requests.post(
    'http://localhost:8000/api/v1/data/edgar/ingest',
    json={
        'tickers': tickers,
        'num_filings': 2
    }
)

print(response.json())
```

### Example 3: Use in Python Scripts

```python
from app.services.data_ingestion.edgar.edgar_ingestion import edgar_ingestion

# Ingest for a single company
result = edgar_ingestion.ingest_company_filings(
    ticker='AAPL',
    num_filings=5
)

print(f"Success: {result['success']}")
print(f"Chunks created: {result['chunks_created']}")
```

## Filing Sections Extracted

The parser extracts key sections from filings:

### 10-K Sections
- **Item 1**: Business
- **Item 1A**: Risk Factors
- **Item 2**: Properties
- **Item 3**: Legal Proceedings
- **Item 6**: Selected Financial Data
- **Item 7**: Management's Discussion and Analysis (MD&A)
- **Item 8**: Financial Statements

### 10-Q Sections
- **Part I**: Financial Information
- **Item 1**: Financial Statements
- **Item 2**: Management's Discussion and Analysis (MD&A)
- **Item 3**: Quantitative and Qualitative Disclosures
- **Item 4**: Controls and Procedures

## RAG Integration

Ingested documents are automatically:
1. **Parsed**: HTML cleaned and text extracted
2. **Chunked**: Split into 1000-character chunks with 200-character overlap
3. **Embedded**: Converted to vector embeddings
4. **Stored**: Added to ChromaDB vector store with metadata

### Metadata Stored

Each chunk includes:
```python
{
    'ticker': 'AAPL',
    'company': 'Apple Inc.',
    'cik': '0000320193',
    'document_type': '10-K',
    'filing_date': '2023-11-03',
    'report_date': '2023-09-30',
    'accession_number': '0000320193-23-000106',
    'source': 'SEC EDGAR',
    'ingestion_date': '2024-01-15T10:30:00',
    'chunk_index': 42,
    'start_char': 42000,
    'end_char': 43000
}
```

## Rate Limiting

The system automatically enforces SEC's rate limit of **10 requests per second**. This is handled transparently in the `EDGARClient` class.

## Best Practices

### 1. Start Small
```bash
# Test with a few companies first
curl -X POST "http://localhost:8000/api/v1/data/edgar/ingest/sample?sample_size=3&num_filings=1"
```

### 2. Batch Processing
For large datasets, process in batches of 10-20 companies:
```python
tickers = sp500_companies.get_tickers()

# Process in batches
batch_size = 10
for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    result = edgar_ingestion.batch_ingest_companies(batch, num_filings=2)
    print(f"Batch {i//batch_size + 1}: {result['total_chunks']} chunks")
```

### 3. Focus on Recent Filings
Limit to recent filings for most up-to-date information:
```python
result = edgar_ingestion.ingest_company_filings(
    ticker='AAPL',
    num_filings=3  # Last 3 filings only
)
```

### 4. Monitor Storage
Check vector store size:
```bash
curl http://localhost:8000/api/v1/chat/stats
```

## Querying Ingested Data

Once data is ingested, use the chat interface with RAG enabled:

```python
import requests

response = requests.post(
    'http://localhost:8000/api/v1/chat/',
    json={
        'query': 'What were Apple\'s main risk factors in their latest 10-K?',
        'use_rag': True,
        'company_filter': 'AAPL'
    }
)

print(response.json()['response'])
```

## Troubleshooting

### Issue: CIK Not Found
**Solution**: The ticker may not be in the S&P 500 list. Use the direct CIK lookup endpoint.

### Issue: Rate Limiting
**Solution**: The system automatically handles rate limiting. If you see delays, this is expected behavior.

### Issue: Large Filing Timeout
**Solution**: Some 10-K filings are very large (50MB+). Increase timeout or limit to 10-Q only.

### Issue: Parsing Errors
**Solution**: Some older filings use different formats. The parser includes fallback to raw text extraction.

## Performance Metrics

Typical performance (on CPU):
- **CIK Lookup**: ~200ms
- **Filing List**: ~300ms
- **Download Filing**: ~1-3 seconds
- **Parse Filing**: ~2-5 seconds
- **Chunk & Embed**: ~10-30 seconds per filing
- **Total per Company** (3 filings): ~2-5 minutes

## Next Steps

After ingesting EDGAR data:
1. Test RAG queries in the chat interface
2. Verify data in the dashboard
3. Add earnings call transcripts (next phase)
4. Set up periodic updates for new filings

## Additional Resources

- [SEC EDGAR API Documentation](https://www.sec.gov/edgar/sec-api-documentation)
- [Understanding SEC Filings](https://www.investor.gov/introduction-investing/investing-basics/glossary/form-10-k)
- [10-K vs 10-Q](https://www.investopedia.com/ask/answers/09/10k-10q-forms.asp)
