from fastapi import APIRouter, HTTPException
from app.models.schemas import MetricRequest, MetricsResponse
from app.services.metrics.metrics_service import metrics_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=MetricsResponse)
async def get_metrics(request: MetricRequest):
    """
    Get financial metrics for a company

    Args:
        request: MetricRequest with company and metric names

    Returns:
        MetricsResponse with time series data
    """
    try:
        result = metrics_service.get_metrics(
            company=request.company,
            metric_names=request.metric_names,
            start_date=request.start_date,
            end_date=request.end_date
        )

        return MetricsResponse(**result)

    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/companies")
async def get_available_companies():
    """Get list of available companies"""
    try:
        companies = metrics_service.get_available_companies()
        return {
            "companies": companies,
            "count": len(companies)
        }
    except Exception as e:
        logger.error(f"Error getting companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available")
async def get_available_metrics():
    """Get list of available metrics"""
    try:
        metrics = metrics_service.get_available_metrics()
        return {
            "metrics": metrics,
            "count": len(metrics)
        }
    except Exception as e:
        logger.error(f"Error getting available metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{company}/{metric_name}")
async def get_latest_metric(company: str, metric_name: str):
    """
    Get the latest value for a specific metric

    Args:
        company: Company ticker
        metric_name: Name of the metric

    Returns:
        Latest metric value
    """
    try:
        value = metrics_service.get_latest_value(company, metric_name)

        if value is None:
            raise HTTPException(
                status_code=404,
                detail=f"Metric '{metric_name}' not found for company '{company}'"
            )

        return {
            "company": company.upper(),
            "metric_name": metric_name,
            "latest_value": value
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))
