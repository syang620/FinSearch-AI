from fastapi import APIRouter, HTTPException
from app.models.schemas import SentimentRequest, SentimentResponse, SentimentScore
from app.services.sentiment.sentiment_service import sentiment_service
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment of text

    Args:
        request: SentimentRequest with text and optional company

    Returns:
        SentimentResponse with sentiment analysis results
    """
    try:
        # Analyze sentiment
        result = sentiment_service.analyze(request.text)

        # Create text snippet (first 200 chars)
        text_snippet = request.text[:200] if len(request.text) > 200 else request.text

        return SentimentResponse(
            sentiment=SentimentScore(
                label=result["label"],
                score=result["score"]
            ),
            text_snippet=text_snippet,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-document/{document_id}")
async def analyze_document_sentiment(document_id: str):
    """
    Analyze sentiment of an entire document

    Args:
        document_id: Document ID to analyze

    Returns:
        Aggregated sentiment for the document
    """
    try:
        # In a real implementation, you would:
        # 1. Retrieve document chunks from vector store
        # 2. Analyze sentiment of each chunk
        # 3. Aggregate results

        # For now, return a placeholder
        return {
            "document_id": document_id,
            "message": "Document sentiment analysis not yet implemented",
            "status": "pending"
        }

    except Exception as e:
        logger.error(f"Error analyzing document sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/company/{company}")
async def get_company_sentiment(company: str):
    """
    Get aggregated sentiment for a company

    Args:
        company: Company name or ticker

    Returns:
        Aggregated sentiment data for the company
    """
    try:
        # In a real implementation, you would:
        # 1. Query all documents related to the company
        # 2. Analyze sentiment across all documents
        # 3. Generate time series data

        # For now, return mock data
        from datetime import timedelta
        import random

        sentiment_data = []
        for i in range(30):  # Last 30 days
            date = datetime.now() - timedelta(days=29 - i)
            label_choices = ["positive", "negative", "neutral"]
            weights = [0.5, 0.2, 0.3]  # More positive sentiment
            label = random.choices(label_choices, weights=weights)[0]

            sentiment_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "label": label,
                "score": random.uniform(0.6, 0.95)
            })

        # Calculate average
        positive_count = sum(1 for s in sentiment_data if s["label"] == "positive")
        negative_count = sum(1 for s in sentiment_data if s["label"] == "negative")
        neutral_count = sum(1 for s in sentiment_data if s["label"] == "neutral")

        if positive_count > negative_count:
            avg_label = "positive"
        elif negative_count > positive_count:
            avg_label = "negative"
        else:
            avg_label = "neutral"

        avg_score = sum(s["score"] for s in sentiment_data) / len(sentiment_data)

        return {
            "company": company,
            "sentiment_data": sentiment_data,
            "average_sentiment": {
                "label": avg_label,
                "score": avg_score
            },
            "breakdown": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count
            }
        }

    except Exception as e:
        logger.error(f"Error getting company sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))
