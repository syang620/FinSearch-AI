"""
S&P 500 Company List Management

Fetches and manages the list of S&P 500 companies with their tickers and CIK numbers.
"""

import pandas as pd
import requests
from typing import List, Dict, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SP500Companies:
    """Manage S&P 500 company data"""

    def __init__(self, cache_file: str = "./data/sp500_companies.json"):
        self.cache_file = Path(cache_file)
        self.companies: List[Dict] = []
        self._load_or_fetch()

    def _load_or_fetch(self):
        """Load from cache or fetch from source"""
        if self.cache_file.exists():
            logger.info("Loading S&P 500 companies from cache")
            with open(self.cache_file, 'r') as f:
                self.companies = json.load(f)
        else:
            logger.info("Fetching S&P 500 companies from Wikipedia")
            self._fetch_from_wikipedia()
            self._save_cache()

    def _fetch_from_wikipedia(self):
        """Fetch S&P 500 list from Wikipedia"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

            # Read tables from Wikipedia
            tables = pd.read_html(url)
            df = tables[0]  # First table contains the S&P 500 list

            # Extract relevant columns
            df = df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
            df.columns = ['ticker', 'company_name', 'sector', 'industry']

            # Convert to list of dicts
            self.companies = df.to_dict('records')

            # Add CIK placeholder (will be fetched separately)
            for company in self.companies:
                company['cik'] = None

            logger.info(f"Fetched {len(self.companies)} S&P 500 companies")

        except Exception as e:
            logger.error(f"Error fetching S&P 500 companies: {e}")
            # Fallback to a small subset for testing
            self.companies = self._get_fallback_companies()

    def _get_fallback_companies(self) -> List[Dict]:
        """Fallback list of major companies"""
        return [
            {"ticker": "AAPL", "company_name": "Apple Inc.", "sector": "Information Technology", "industry": "Technology Hardware, Storage & Peripherals", "cik": "0000320193"},
            {"ticker": "MSFT", "company_name": "Microsoft Corporation", "sector": "Information Technology", "industry": "Systems Software", "cik": "0000789019"},
            {"ticker": "GOOGL", "company_name": "Alphabet Inc.", "sector": "Communication Services", "industry": "Interactive Media & Services", "cik": "0001652044"},
            {"ticker": "AMZN", "company_name": "Amazon.com Inc.", "sector": "Consumer Discretionary", "industry": "Internet & Direct Marketing Retail", "cik": "0001018724"},
            {"ticker": "TSLA", "company_name": "Tesla, Inc.", "sector": "Consumer Discretionary", "industry": "Automobile Manufacturers", "cik": "0001318605"},
            {"ticker": "META", "company_name": "Meta Platforms Inc.", "sector": "Communication Services", "industry": "Interactive Media & Services", "cik": "0001326801"},
            {"ticker": "NVDA", "company_name": "NVIDIA Corporation", "sector": "Information Technology", "industry": "Semiconductors", "cik": "0001045810"},
            {"ticker": "JPM", "company_name": "JPMorgan Chase & Co.", "sector": "Financials", "industry": "Diversified Banks", "cik": "0000019617"},
            {"ticker": "V", "company_name": "Visa Inc.", "sector": "Information Technology", "industry": "Data Processing & Outsourced Services", "cik": "0001403161"},
            {"ticker": "JNJ", "company_name": "Johnson & Johnson", "sector": "Health Care", "industry": "Pharmaceuticals", "cik": "0000200406"},
        ]

    def _save_cache(self):
        """Save companies to cache file"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.companies, f, indent=2)
            logger.info(f"Saved S&P 500 companies to {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def get_all_companies(self) -> List[Dict]:
        """Get all S&P 500 companies"""
        return self.companies

    def get_company_by_ticker(self, ticker: str) -> Optional[Dict]:
        """Get company by ticker symbol"""
        ticker = ticker.upper()
        for company in self.companies:
            if company['ticker'] == ticker:
                return company
        return None

    def get_companies_by_sector(self, sector: str) -> List[Dict]:
        """Get companies by sector"""
        return [c for c in self.companies if c['sector'] == sector]

    def get_sectors(self) -> List[str]:
        """Get list of unique sectors"""
        return list(set(c['sector'] for c in self.companies))

    def update_cik(self, ticker: str, cik: str):
        """Update CIK for a company"""
        company = self.get_company_by_ticker(ticker)
        if company:
            company['cik'] = cik
            self._save_cache()
            logger.info(f"Updated CIK for {ticker}: {cik}")

    def get_tickers(self) -> List[str]:
        """Get list of all tickers"""
        return [c['ticker'] for c in self.companies]


# Singleton instance
sp500_companies = SP500Companies()
