from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import random

logger = logging.getLogger(__name__)


class MetricsService:
    """Service for managing financial metrics data"""

    def __init__(self):
        # In a real application, this would connect to a database or financial API
        # For now, we'll use mock data
        self.mock_data = self._generate_mock_data()

    def _generate_mock_data(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Generate mock financial metrics data for demonstration"""
        companies = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        metrics = ["revenue", "eps", "profit_margin", "operating_income", "free_cash_flow"]

        data = {}
        for company in companies:
            data[company] = {}
            for metric in metrics:
                # Generate time series data for the last 8 quarters
                data[company][metric] = []
                base_value = random.uniform(10, 100)

                for i in range(8):
                    # Calculate date (quarterly data)
                    date = datetime.now() - timedelta(days=90 * (7 - i))
                    # Add some growth trend with randomness
                    value = base_value * (1 + 0.05 * i) * random.uniform(0.9, 1.1)

                    data[company][metric].append({
                        "date": date.strftime("%Y-%m-%d"),
                        "value": round(value, 2),
                        "metric_name": metric
                    })

        return data

    def get_metrics(
        self,
        company: str,
        metric_names: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get financial metrics for a company

        Args:
            company: Company ticker symbol
            metric_names: List of metric names to retrieve
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            Dictionary with metrics data
        """
        try:
            company_upper = company.upper()

            if company_upper not in self.mock_data:
                logger.warning(f"Company {company_upper} not found in mock data")
                return {
                    "company": company_upper,
                    "metrics": {},
                    "metadata": {"message": "Company not found"}
                }

            # Retrieve requested metrics
            result_metrics = {}
            for metric_name in metric_names:
                if metric_name in self.mock_data[company_upper]:
                    metric_data = self.mock_data[company_upper][metric_name]

                    # Filter by date if provided
                    if start_date or end_date:
                        filtered_data = []
                        for point in metric_data:
                            point_date = point["date"]
                            if start_date and point_date < start_date:
                                continue
                            if end_date and point_date > end_date:
                                continue
                            filtered_data.append(point)
                        result_metrics[metric_name] = filtered_data
                    else:
                        result_metrics[metric_name] = metric_data
                else:
                    logger.warning(f"Metric {metric_name} not found for {company_upper}")
                    result_metrics[metric_name] = []

            return {
                "company": company_upper,
                "metrics": result_metrics,
                "metadata": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "data_points": sum(len(v) for v in result_metrics.values())
                }
            }

        except Exception as e:
            logger.error(f"Error retrieving metrics: {e}")
            raise

    def get_available_companies(self) -> List[str]:
        """Get list of available companies"""
        return list(self.mock_data.keys())

    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics"""
        if self.mock_data:
            first_company = next(iter(self.mock_data.keys()))
            return list(self.mock_data[first_company].keys())
        return []

    def get_latest_value(self, company: str, metric_name: str) -> Optional[float]:
        """Get the latest value for a specific metric"""
        try:
            company_upper = company.upper()
            if company_upper in self.mock_data and metric_name in self.mock_data[company_upper]:
                data = self.mock_data[company_upper][metric_name]
                if data:
                    return data[-1]["value"]
            return None
        except Exception as e:
            logger.error(f"Error getting latest value: {e}")
            return None


# Singleton instance
metrics_service = MetricsService()
