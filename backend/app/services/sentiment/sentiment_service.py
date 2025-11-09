from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import Dict, Any, List
from app.core.config import settings
import torch
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SentimentService:
    """Service for financial sentiment analysis using FinBERT"""

    def __init__(self):
        self.model_name = settings.SENTIMENT_MODEL
        self.device = settings.LLM_DEVICE
        self.pipeline = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the FinBERT model"""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")

            # Create sentiment analysis pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1
            )

            logger.info("Sentiment model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment label and score
        """
        try:
            # Truncate text if too long (FinBERT max is 512 tokens)
            max_chars = 2000
            if len(text) > max_chars:
                text = text[:max_chars]

            # Run sentiment analysis
            result = self.pipeline(text)[0]

            # Normalize label (FinBERT outputs: positive, negative, neutral)
            label = result['label'].lower()

            return {
                "label": label,
                "score": float(result['score']),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            raise

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple texts

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment results
        """
        try:
            # Truncate texts if needed
            max_chars = 2000
            truncated_texts = [text[:max_chars] if len(text) > max_chars else text for text in texts]

            # Run batch sentiment analysis
            results = self.pipeline(truncated_texts)

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "label": result['label'].lower(),
                    "score": float(result['score']),
                    "timestamp": datetime.now().isoformat()
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error analyzing batch sentiment: {e}")
            raise

    def get_aggregated_sentiment(self, sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple sentiment scores

        Args:
            sentiments: List of sentiment results

        Returns:
            Aggregated sentiment
        """
        if not sentiments:
            return {
                "label": "neutral",
                "score": 0.0,
                "count": 0
            }

        # Count sentiments
        sentiment_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }

        total_score = 0
        for sentiment in sentiments:
            label = sentiment["label"]
            score = sentiment["score"]
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
            # Weight score by sentiment type
            if label == "positive":
                total_score += score
            elif label == "negative":
                total_score -= score

        # Determine overall sentiment
        if sentiment_counts["positive"] > sentiment_counts["negative"]:
            overall_label = "positive"
        elif sentiment_counts["negative"] > sentiment_counts["positive"]:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        # Calculate average score
        avg_score = abs(total_score) / len(sentiments) if sentiments else 0

        return {
            "label": overall_label,
            "score": avg_score,
            "count": len(sentiments),
            "breakdown": sentiment_counts
        }


# Singleton instance
sentiment_service = SentimentService()
