"""
Top 10 S&P 500 Companies by Market Cap (MVP)

Hardcoded list of top 10 companies for MVP scope.
Market cap data as of early 2024.
"""

from typing import List, Dict

# Top 10 S&P 500 companies by market capitalization
TOP_10_COMPANIES = [
    {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Information Technology",
        "market_cap_b": 3000,  # Approximate market cap in billions
    },
    {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "sector": "Information Technology",
        "market_cap_b": 2800,
    },
    {
        "ticker": "GOOGL",
        "company_name": "Alphabet Inc.",
        "sector": "Communication Services",
        "market_cap_b": 1750,
    },
    {
        "ticker": "AMZN",
        "company_name": "Amazon.com Inc.",
        "sector": "Consumer Discretionary",
        "market_cap_b": 1650,
    },
    {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "sector": "Information Technology",
        "market_cap_b": 1500,
    },
    {
        "ticker": "META",
        "company_name": "Meta Platforms Inc.",
        "sector": "Communication Services",
        "market_cap_b": 900,
    },
    {
        "ticker": "TSLA",
        "company_name": "Tesla, Inc.",
        "sector": "Consumer Discretionary",
        "market_cap_b": 850,
    },
    {
        "ticker": "BRK.B",
        "company_name": "Berkshire Hathaway Inc.",
        "sector": "Financials",
        "market_cap_b": 850,
    },
    {
        "ticker": "V",
        "company_name": "Visa Inc.",
        "sector": "Information Technology",
        "market_cap_b": 550,
    },
    {
        "ticker": "UNH",
        "company_name": "UnitedHealth Group Inc.",
        "sector": "Health Care",
        "market_cap_b": 500,
    },
]


def get_top_companies() -> List[Dict]:
    """Get list of top 10 companies"""
    return TOP_10_COMPANIES


def get_top_tickers() -> List[str]:
    """Get list of top 10 tickers"""
    return [company["ticker"] for company in TOP_10_COMPANIES]


def get_company_by_ticker(ticker: str) -> Dict:
    """Get company info by ticker"""
    for company in TOP_10_COMPANIES:
        if company["ticker"] == ticker:
            return company
    return None


# MVP scope: 8 quarters = 2 years
QUARTERS_TO_FETCH = 8
