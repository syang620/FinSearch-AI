from sentence_transformers import SentenceTransformer
from typing import List
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers"""

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text string

        Returns:
            List of floats representing the embedding
        """
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of text strings

        Returns:
            List of embeddings
        """
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


# Singleton instance
embedding_service = EmbeddingService()
