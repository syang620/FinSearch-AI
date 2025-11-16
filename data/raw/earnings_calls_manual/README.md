# Manual Earnings Call Transcripts

This folder contains manually downloaded earnings call transcripts for the top 10 S&P 500 companies.

## Folder Structure

```
data/earnings_calls_manual/
├── AAPL/          # Apple Inc.
├── MSFT/          # Microsoft Corporation
├── GOOGL/         # Alphabet Inc.
├── AMZN/          # Amazon.com Inc.
├── NVDA/          # NVIDIA Corporation
├── META/          # Meta Platforms Inc.
├── TSLA/          # Tesla Inc.
├── BRK.B/         # Berkshire Hathaway Inc.
├── V/             # Visa Inc.
└── JPM/           # JPMorgan Chase & Co.
```

## File Naming Convention

Please use this naming format for each file:

**Format:** `{TICKER}_Q{QUARTER}_{YEAR}.{ext}`

**Examples:**
- `AAPL_Q4_2024.pdf`
- `AAPL_Q4_2024.txt`
- `MSFT_Q3_2024.pdf`
- `GOOGL_Q2_2023.txt`

## Quarters to Download

For each company, please download **8 most recent quarters** (approximately 2 years):

### Target Quarters (as of Nov 2024):

**2024:**
- Q4 2024 (Oct-Dec)
- Q3 2024 (Jul-Sep)
- Q2 2024 (Apr-Jun)
- Q1 2024 (Jan-Mar)

**2023:**
- Q4 2023 (Oct-Dec)
- Q3 2023 (Jul-Sep)
- Q2 2023 (Apr-Jun)
- Q1 2023 (Jan-Mar)

**Note:** If Q4 2024 isn't available yet, go back one more quarter (Q4 2022).

## Supported File Formats

- **PDF** (`.pdf`) - Preferred
- **TXT** (`.txt`) - Plain text also works
- **DOCX** (`.docx`) - Word documents

## Where to Download From

You can download transcripts from:

1. **discountingcashflows.com** (you have an account)
   - Navigate to: `https://discountingcashflows.com/company/{TICKER}/transcripts/`
   - Select the quarter
   - Download or copy the text

2. **Seeking Alpha** (may require subscription)
   - Search for "{TICKER} earnings call transcript Q{X} {YEAR}"

3. **Company Investor Relations**
   - AAPL: https://investor.apple.com/
   - MSFT: https://www.microsoft.com/en-us/investor
   - etc.

## Download Checklist

- [ ] AAPL - 8 quarters
- [ ] MSFT - 8 quarters
- [ ] GOOGL - 8 quarters
- [ ] AMZN - 8 quarters
- [ ] NVDA - 8 quarters
- [ ] META - 8 quarters
- [ ] TSLA - 8 quarters
- [ ] BRK.B - 8 quarters
- [ ] V - 8 quarters
- [ ] JPM - 8 quarters

**Total:** 80 transcript files

## After Download

Once you've downloaded all files, run:

```bash
python ingest_manual_transcripts.py
```

This script will:
1. Parse all PDF/TXT files
2. Extract transcript text
3. Store in the database
4. Prepare for RAG ingestion

## Example File Structure (After Download)

```
data/earnings_calls_manual/
├── AAPL/
│   ├── AAPL_Q1_2023.pdf
│   ├── AAPL_Q2_2023.pdf
│   ├── AAPL_Q3_2023.pdf
│   ├── AAPL_Q4_2023.pdf
│   ├── AAPL_Q1_2024.pdf
│   ├── AAPL_Q2_2024.pdf
│   ├── AAPL_Q3_2024.pdf
│   └── AAPL_Q4_2024.pdf
├── MSFT/
│   ├── MSFT_Q1_2023.txt
│   ├── MSFT_Q2_2023.txt
│   └── ...
└── ...
```

## Tips

1. **Consistent naming:** Stick to the naming convention for easier processing
2. **One file per quarter:** Don't combine multiple quarters in one file
3. **Check file size:** Transcripts are usually 100KB - 1MB
4. **Verify content:** Ensure files contain actual transcript text, not just summaries
5. **Save both formats:** If available, save both PDF and TXT for backup

## Need Help?

If you have questions about:
- Which quarters to download
- File naming
- Where to find transcripts
- Processing the files

Just ask!
