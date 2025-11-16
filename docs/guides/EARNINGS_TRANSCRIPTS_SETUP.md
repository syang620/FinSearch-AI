# Earnings Call Transcripts Setup

This guide explains how to fetch earnings call transcripts from discountingcashflows.com using the authenticated scraper.

## Prerequisites

1. **Account**: Create a free account at [discountingcashflows.com](https://discountingcashflows.com/accounts/signup/)
2. **Python Dependencies**: Install required packages

```bash
cd backend
pip install -r requirements.txt
```

## Configuration

### Step 1: Add Your Credentials

Edit the `.env` file in the project root and add your DCF credentials:

```bash
# .env file
DCF_EMAIL=your_email@example.com
DCF_PASSWORD=your_secure_password
```

**Security Notes:**
- The `.env` file is gitignored and will not be committed
- Session cookies are cached in `~/.dcf_cookies.pkl` (also gitignored)
- You can revoke access anytime by changing your password

### Step 2: Verify Configuration

The `.env.example` file shows all available configuration options:

```bash
cat .env.example
```

## Testing

### Quick Test with AAPL

Run the test script to verify everything works:

```bash
python test_dcf.py
```

This will:
1. Authenticate with discountingcashflows.com
2. Try fetching AAPL transcripts for Q4 2024, Q3 2024, and Q4 2025
3. Save the first successful transcript to a file
4. Display a preview

**Expected Output:**
```
======================================================================
DCF (discountingcashflows.com) Transcript Fetcher Test
======================================================================

Using credentials: you***@example.com

Testing ticker: AAPL

======================================================================
Testing AAPL Q4 2024
======================================================================

✓ Successfully fetched transcript!

Metadata:
  Ticker: AAPL
  Company: Apple Inc.
  Title: Apple Inc. Q4 2024 Earnings Call Transcript
  Date: 2024-11-01
  Year: 2024
  Quarter: Q4
  Participants: 15
  Transcript length: 45823 chars

----------------------------------------------------------------------
Transcript Preview (first 1000 characters):
----------------------------------------------------------------------
[Transcript content preview...]

✓ Saved to: earnings_dcf_AAPL_Q4_2024.txt
```

## Usage in Code

### Fetch Single Transcript

```python
from app.services.data_ingestion.earnings.earnings_fetcher import earnings_fetcher

# Fetch specific quarter
transcript = earnings_fetcher.fetch_transcript(
    ticker="AAPL",
    year=2024,
    quarter=4
)

if transcript:
    print(f"Ticker: {transcript['ticker']}")
    print(f"Title: {transcript['title']}")
    print(f"Transcript: {transcript['transcript'][:500]}...")
```

### Fetch Multiple Recent Transcripts

```python
# Fetch last 8 quarters
transcripts = earnings_fetcher.fetch_recent_transcripts(
    ticker="AAPL",
    num_quarters=8
)

print(f"Fetched {len(transcripts)} transcripts")
for t in transcripts:
    print(f"  - Q{t['quarter']} {t['year']}")
```

### Save to File Storage

```python
from app.services.data_ingestion.file_storage import file_storage

transcript = earnings_fetcher.fetch_transcript("AAPL", 2024, 4)

if transcript:
    file_storage.save_earnings_call(
        ticker=transcript['ticker'],
        year=transcript['year'],
        quarter=transcript['quarter'],
        transcript=transcript['transcript'],
        metadata={
            'title': transcript['title'],
            'date': transcript['publish_date'],
            'participants': transcript['participants'],
            'source': 'discountingcashflows.com'
        }
    )
```

## Architecture

### Module Overview

```
backend/app/services/data_ingestion/earnings/
├── __init__.py              # Module exports
├── dcf_auth.py             # Authentication & session management
├── dcf_scraper.py          # HTML scraping logic
└── earnings_fetcher.py     # Main orchestration layer
```

### How It Works

1. **Authentication** (`dcf_auth.py`):
   - Logs into discountingcashflows.com
   - Handles CSRF tokens
   - Caches session cookies for reuse
   - Verifies session validity

2. **Scraping** (`dcf_scraper.py`):
   - Uses authenticated session
   - Fetches transcript pages
   - Parses HTML to extract text
   - Handles multiple HTML structures

3. **Orchestration** (`earnings_fetcher.py`):
   - Provides clean API
   - Manages quarter/year logic
   - Returns structured data

## Transcript Data Structure

Each transcript is returned as a dictionary:

```python
{
    'ticker': 'AAPL',
    'company_name': 'Apple Inc.',
    'title': 'Apple Inc. Q4 2024 Earnings Call Transcript',
    'publish_date': '2024-11-01',
    'year': 2024,
    'quarter': 4,
    'transcript': '...',  # Full transcript text
    'participants': ['Tim Cook', 'Luca Maestri', ...],
    'url': 'https://discountingcashflows.com/company/AAPL/transcripts/2024/4/'
}
```

## Troubleshooting

### Authentication Failed

**Error:** `Failed to authenticate` or `Not authenticated - redirected to login`

**Solutions:**
1. Verify credentials in `.env` file
2. Try logging in manually at discountingcashflows.com
3. Delete cached session: `rm ~/.dcf_cookies.pkl`
4. Check if account is active

### No Transcript Found

**Error:** `No transcript found for AAPL Q4 2024`

**Solutions:**
1. Verify transcript exists on the website
2. Try a different quarter
3. Check if ticker is correct (some companies use different symbols)

### Empty Transcript Text

**Error:** `No transcript text found on page`

**Solutions:**
1. Website HTML structure may have changed
2. Check `dcf_scraper.py` extraction logic
3. Inspect page source manually
4. May need to update CSS selectors

### Rate Limiting

DCF may rate limit excessive requests. The scraper includes built-in delays but:
- Wait 5 seconds between requests
- Don't fetch too many transcripts in short time
- Use cached sessions to reduce login requests

## MVP Data Collection

For the MVP (Top 10 S&P 500 companies, 8 quarters each):

```python
# Example: Fetch all data for MVP
from app.services.data_ingestion.top_companies import TOP_10_COMPANIES

for company in TOP_10_COMPANIES:
    ticker = company['ticker']
    transcripts = earnings_fetcher.fetch_recent_transcripts(ticker, num_quarters=8)

    for transcript in transcripts:
        # Save each transcript
        file_storage.save_earnings_call(...)
```

**Estimated Time:** ~10-15 minutes for 80 transcripts (with rate limiting)

## Alternative: Manual PDF Download

If scraping doesn't work reliably, you can manually download PDFs:

1. Visit discountingcashflows.com manually
2. Navigate to each company's transcript page
3. Download PDF for each quarter
4. Use our PDF parser (to be implemented)

This is slower but more reliable for MVP.

## Security & Ethics

- **Authentication**: Uses legitimate login credentials (not bypassing authentication)
- **Rate Limiting**: Respects server load with delays
- **Terms of Service**: Review DCF's ToS for acceptable use
- **Data Usage**: Only for personal/research purposes in this MVP

## Next Steps

1. **Test the scraper** with your credentials
2. **Fetch sample data** for 1-2 companies
3. **Inspect quality** of transcript text
4. **Decide approach**:
   - Continue with automated scraping, or
   - Switch to manual PDF download

## Support

For issues:
1. Check logs for detailed error messages
2. Verify website hasn't changed structure
3. Try manual login to confirm account works
4. Consider alternative data sources if needed
