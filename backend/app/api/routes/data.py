"""
Data Ingestion API Routes

Endpoints for managing data ingestion from SEC EDGAR and other sources.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from app.services.data_ingestion.edgar.sp500_companies import sp500_companies
from app.services.data_ingestion.edgar.edgar_client import edgar_client
from app.services.data_ingestion.edgar.edgar_ingestion import edgar_ingestion

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response Models
class IngestionRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of stock tickers to ingest")
    num_filings: int = Field(3, description="Number of recent filings per company", ge=1, le=10)
    form_types: Optional[List[str]] = Field(['10-K', '10-Q'], description="Form types to ingest")


class IngestionResponse(BaseModel):
    success: bool
    message: str
    companies_processed: int
    total_filings: int
    total_chunks: int
    details: Optional[dict] = None


class FilingListResponse(BaseModel):
    ticker: str
    cik: str
    filings: List[dict]
    count: int


# Ingestion state tracking
ingestion_status = {
    'is_running': False,
    'current_task': None,
    'last_result': None
}


@router.get("/companies")
async def get_sp500_companies():
    """
    Get list of S&P 500 companies

    Returns:
        List of company information
    """
    try:
        companies = sp500_companies.get_all_companies()

        return {
            'success': True,
            'count': len(companies),
            'companies': companies
        }

    except Exception as e:
        logger.error(f"Error getting companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/companies/sectors")
async def get_sectors():
    """Get list of unique sectors"""
    try:
        sectors = sp500_companies.get_sectors()

        return {
            'success': True,
            'count': len(sectors),
            'sectors': sorted(sectors)
        }

    except Exception as e:
        logger.error(f"Error getting sectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/companies/sector/{sector}")
async def get_companies_by_sector(sector: str):
    """Get companies in a specific sector"""
    try:
        companies = sp500_companies.get_companies_by_sector(sector)

        return {
            'success': True,
            'sector': sector,
            'count': len(companies),
            'companies': companies
        }

    except Exception as e:
        logger.error(f"Error getting companies by sector: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/edgar/cik/{ticker}")
async def get_cik(ticker: str):
    """
    Get CIK number for a ticker

    Args:
        ticker: Stock ticker symbol

    Returns:
        CIK number and company info
    """
    try:
        ticker = ticker.upper()

        # Get from EDGAR
        cik = edgar_client.get_cik_by_ticker(ticker)

        if not cik:
            raise HTTPException(status_code=404, detail=f"CIK not found for ticker {ticker}")

        # Get company info
        company = sp500_companies.get_company_by_ticker(ticker)

        return {
            'ticker': ticker,
            'cik': cik,
            'company_info': company
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting CIK: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/edgar/filings/{ticker}", response_model=FilingListResponse)
async def get_company_filings(
    ticker: str,
    form_types: Optional[str] = None,
    limit: int = 10
):
    """
    Get list of filings for a company

    Args:
        ticker: Stock ticker symbol
        form_types: Comma-separated form types (e.g., '10-K,10-Q')
        limit: Maximum number of filings to return

    Returns:
        List of filings
    """
    try:
        ticker = ticker.upper()

        # Get CIK
        cik = edgar_client.get_cik_by_ticker(ticker)
        if not cik:
            raise HTTPException(status_code=404, detail=f"CIK not found for ticker {ticker}")

        # Parse form types
        form_type_list = None
        if form_types:
            form_type_list = [ft.strip() for ft in form_types.split(',')]

        # Get filings
        filings = edgar_client.get_company_filings(cik, form_types=form_type_list)

        # Limit results
        filings = filings[:limit]

        return FilingListResponse(
            ticker=ticker,
            cik=cik,
            filings=filings,
            count=len(filings)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting filings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edgar/ingest", response_model=IngestionResponse)
async def ingest_edgar_data(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest SEC EDGAR filings for specified companies

    Args:
        request: Ingestion request with tickers and options

    Returns:
        Ingestion results
    """
    try:
        # Check if ingestion is already running
        if ingestion_status['is_running']:
            raise HTTPException(
                status_code=409,
                detail="Ingestion already in progress"
            )

        logger.info(f"Starting ingestion for {len(request.tickers)} companies")

        # For now, run synchronously
        # In production, you'd want to use background tasks or a task queue
        result = edgar_ingestion.batch_ingest_companies(
            tickers=request.tickers,
            num_filings=request.num_filings
        )

        return IngestionResponse(
            success=result['success'],
            message=f"Ingested {result['total_filings']} filings for {result['companies_processed']} companies",
            companies_processed=result['companies_processed'],
            total_filings=result['total_filings'],
            total_chunks=result['total_chunks'],
            details=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/edgar/ingest/sample")
async def ingest_sp500_sample(sample_size: int = 5, num_filings: int = 2):
    """
    Ingest sample of S&P 500 companies

    Args:
        sample_size: Number of companies to sample
        num_filings: Number of filings per company

    Returns:
        Ingestion results
    """
    try:
        logger.info(f"Starting sample ingestion: {sample_size} companies")

        result = edgar_ingestion.ingest_sp500_sample(
            sample_size=sample_size,
            num_filings=num_filings
        )

        return IngestionResponse(
            success=result['success'],
            message=f"Ingested sample data for {result['companies_processed']} companies",
            companies_processed=result['companies_processed'],
            total_filings=result['total_filings'],
            total_chunks=result['total_chunks'],
            details=result
        )

    except Exception as e:
        logger.error(f"Error in sample ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/edgar/status")
async def get_ingestion_status():
    """Get current ingestion status"""
    return ingestion_status
