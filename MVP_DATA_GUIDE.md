# MVP Data Ingestion Guide

## Overview

This MVP implementation focuses on getting real financial data quickly with a simple, file-based approach.

### Scope
- **Companies**: Top 10 S&P 500 by market cap
- **Time Period**: Past 8 quarters (~2 years)
- **Data Sources**:
  - SEC EDGAR (10-K and 10-Q filings)
  - Earnings Call Transcripts (via earningscall package or mock data)
- **Storage**: Simple filesystem (no database for MVP)

### Top 10 Companies

1. **AAPL** - Apple Inc.
2. **MSFT** - Microsoft Corporation
3. **GOOGL** - Alphabet Inc.
4. **AMZN** - Amazon.com Inc.
5. **NVDA** - NVIDIA Corporation
6. **META** - Meta Platforms Inc.
7. **TSLA** - Tesla, Inc.
8. **BRK.B** - Berkshire Hathaway Inc.
9. **V** - Visa Inc.
10. **UNH** - UnitedHealth Group Inc.

## Architecture

```
┌─────────────────────────────────────────────────┐
│         SEC EDGAR + Earnings Call APIs          │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────┐
        │  MVP Ingestion       │
        │  (ingest_mvp_data.py)│
        └──────────┬───────────┘
                   │
        ┌──────────┴───────────┐
        │                      │
        ↓                      ↓
┌───────────────┐    ┌───────────────┐
│  EDGAR Data   │    │ Earnings Calls│
│  (Download &  │    │  (Fetch       │
│   Parse)      │    │   Transcripts)│
└───────┬───────┘    └───────┬───────┘
        │                    │
        └────────┬───────────┘
                 ↓
        ┌─────────────────┐
        │  File Storage   │
        │  • Raw files    │
        │  • Parsed text  │
        │  • Metadata     │
        └────────┬────────┘
                 │
                 ↓
        ┌─────────────────┐
        │  RAG System     │
        │  (Optional)     │
        └─────────────────┘
```

## Storage Structure

```
data/
├── edgar/
│   ├── AAPL/
│   │   ├── raw_10-K_2024.htm           # 1.5MB - Original HTML from SEC
│   │   ├── 10-K_2024.txt               # 500KB - Cleaned text
│   │   ├── raw_10-Q_2024_Q1.htm
│   │   ├── 10-Q_2024_Q1.txt
│   │   ├── raw_10-Q_2024_Q2.htm
│   │   ├── 10-Q_2024_Q2.txt
│   │   └── metadata.json               # Filing metadata
│   ├── MSFT/
│   │   └── ...
│   └── ...
│
└── earnings_calls/
    ├── AAPL/
    │   ├── 2024_Q1.txt                 # Transcript text
    │   ├── 2024_Q2.txt
    │   ├── 2024_Q3.txt
    │   └── metadata.json               # Call metadata
    ├── MSFT/
    │   └── ...
    └── ...
```

### Metadata Format

**EDGAR metadata.json:**
```json
{
  "ticker": "AAPL",
  "filings": [
    {
      "type": "10-K",
      "date": "2024-11-01",
      "year": "2024",
      "quarter": null,
      "raw_file": "raw_10-K_2024.htm",
      "parsed_file": "10-K_2024.txt",
      "rag_ingested": false,
      "saved_at": "2024-01-15T10:30:00",
      "cik": "0000320193",
      "accession_number": "0000320193-24-000106"
    }
  ]
}
```

**Earnings metadata.json:**
```json
{
  "ticker": "AAPL",
  "earnings_calls": [
    {
      "year": 2024,
      "quarter": 3,
      "file": "2024_Q3.txt",
      "rag_ingested": false,
      "saved_at": "2024-01-15T10:35:00",
      "date": "2024-08-01",
      "participants": ["CEO", "CFO", "Analysts"],
      "is_mock": true
    }
  ]
}
```

## Usage

### Step 1: Run MVP Ingestion

```bash
cd "FinSearch AI"

# Run the ingestion script
python ingest_mvp_data.py
```

**What it does:**
1. Fetches CIKs for top 10 companies
2. Downloads 8 quarters of 10-K/10-Q filings from SEC
3. Parses and cleans the HTML
4. Saves both raw and parsed versions
5. Fetches 8 quarters of earnings call transcripts
6. Saves all data to `data/` directory
7. Creates metadata.json for each company

**Expected output:**
```
======================================================================
FinSearch AI - MVP Data Ingestion
======================================================================

Scope:
  • Top 10 S&P 500 companies by market cap
  • 8 quarters of data (~2 years)
  • EDGAR filings (10-K/10-Q)
  • Earnings call transcripts

Storage: Filesystem (data/edgar/ and data/earnings_calls/)

----------------------------------------------------------------------

[1/10] Processing AAPL...
----------------------------------------------------------------------
=== Ingesting EDGAR data for AAPL ===
Found CIK: 0000320193
Found 8 filings
  ✓ 10-K (2024-11-01) - saved to disk
  ✓ 10-Q (2024-08-01) - saved to disk
  ...

=== Ingesting earnings calls for AAPL ===
Found 8 transcripts
  ✓ Q3 2024 - saved to disk
  ...

[2/10] Processing MSFT...
...

======================================================================
INGESTION COMPLETE
======================================================================

Companies processed: 10/10
EDGAR filings saved: 75
Earnings calls saved: 80

Data saved to:
  • EDGAR filings: ./data/edgar
  • Earnings calls: ./data/earnings_calls
```

### Step 2: Inspect the Data

```bash
# Check what was downloaded
ls -lh data/edgar/AAPL/
ls -lh data/earnings_calls/AAPL/

# Read a parsed filing
cat data/edgar/AAPL/10-K_2024.txt | head -100

# Read an earnings transcript
cat data/earnings_calls/AAPL/2024_Q3.txt
```

### Step 3: Load into RAG (Optional)

After data is saved to disk, you can load it into the vector database:

```python
# Coming in next phase - load_to_rag.py
python load_to_rag.py
```

## Key Features

### 1. **File-Based Storage**
✅ Simple - just files and folders
✅ Easy to inspect and debug
✅ No database setup required
✅ Can version control (if desired)
✅ Easy to delete and re-ingest

### 2. **Reprocessability**
✅ Raw HTML saved for future reprocessing
✅ Can experiment with different parsing strategies
✅ Don't need to re-download from SEC

### 3. **Metadata Tracking**
✅ JSON files track what's been downloaded
✅ Track RAG ingestion status
✅ Easy to query with Python/jq

### 4. **Mock Data Fallback**
✅ Works even if earningscall package not installed
✅ Uses mock transcripts for testing
✅ Real data when package is available

## Estimated Data Size

**Per Company:**
- EDGAR raw files: ~8-12 MB (8 filings × 1.5 MB)
- EDGAR parsed files: ~3-5 MB (8 filings × 500 KB)
- Earnings transcripts: ~200-500 KB (8 calls)
- **Total per company: ~12-18 MB**

**Total for 10 Companies:**
- **~120-180 MB total**

## Troubleshooting

### Issue: CIK not found
**Solution**: Check ticker symbol is correct. Some companies use different tickers.

### Issue: No filings found
**Solution**: Company may not have 8 quarters of data yet (new IPO). Script continues with available filings.

### Issue: SEC rate limiting
**Solution**: Script automatically rate limits to 10 req/sec. Just be patient.

### Issue: Earnings call package not installed
**Solution**: Script uses mock data automatically. For real data:
```bash
pip install git+https://github.com/EarningsCall/earningscall-python.git
```

## Next Steps

1. **Run ingestion**: `python ingest_mvp_data.py`
2. **Inspect files**: Check `data/` directory
3. **Load to RAG**: Run RAG ingestion script (coming next)
4. **Test queries**: Use chat interface to query the data

## File Details

### New Files Created

```
backend/app/services/data_ingestion/
├── top_companies.py              # Top 10 company list
├── file_storage.py               # File storage manager
└── earnings/
    └── earnings_fetcher.py       # Earnings call fetcher

ingest_mvp_data.py                # Main ingestion script
```

### Modified Files

```
backend/app/services/data_ingestion/edgar/
└── edgar_ingestion.py            # Added file storage support

backend/requirements.txt          # Added earnings call package
```

## Benefits of MVP Approach

1. **Fast to implement** - No database complexity
2. **Easy to debug** - Can inspect files manually
3. **Flexible** - Easy to change parsing/chunking later
4. **Reusable** - Data stored for multiple uses
5. **Foundation** - Can add database layer later if needed

## Future Enhancements

After MVP is working:
- [ ] Add SQLite metadata database
- [ ] Automated RAG ingestion
- [ ] Periodic updates for new filings
- [ ] More companies (all S&P 500)
- [ ] Financial news integration
- [ ] Advanced analytics on stored data
