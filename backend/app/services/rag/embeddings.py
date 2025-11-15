from sentence_transformers import SentenceTransformer
from typing import List, Optional
from app.core.config import settings
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers"""

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.expected_dimension = settings.EMBEDDING_DIMENSION
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Verify dimension matches expected
            actual_dim = self.model.get_sentence_embedding_dimension()
            if actual_dim != self.expected_dimension:
                logger.warning(
                    f"Model dimension {actual_dim} != expected {self.expected_dimension}"
                )

            logger.info(f"Embedding model loaded successfully (dim={actual_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _validate_text(self, text: str) -> str:
        """
        Validate and clean input text

        Args:
            text: Input text string

        Returns:
            Cleaned text string
        """
        if text is None:
            logger.warning("Received None text, converting to empty string")
            return ""

        if not isinstance(text, str):
            logger.warning(f"Converting non-string type {type(text)} to string")
            text = str(text)

        # Warn about very long texts (model will truncate)
        if len(text) > 10000:
            logger.warning(f"Text length {len(text)} exceeds 10000 chars, will be truncated by model")

        return text

    def _validate_embedding(self, embedding: np.ndarray, text_snippet: str = "") -> List[float]:
        """
        Validate embedding output

        Args:
            embedding: Numpy array from model
            text_snippet: Text snippet for logging

        Returns:
            List of floats
        """
        # Check for NaN or Inf
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            logger.error(f"Invalid embedding values for text: {text_snippet[:50]}")
            raise ValueError("Embedding contains NaN or Inf values")

        # Verify dimension
        if len(embedding) != self.expected_dimension:
            logger.error(
                f"Unexpected embedding dimension: {len(embedding)} != {self.expected_dimension}"
            )

        return embedding.tolist()

    def embed_text(self, text: str, show_progress: bool = False) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text string
            show_progress: Whether to show progress bar (default: False)

        Returns:
            List of floats representing the embedding
        """
        try:
            # Validate input
            text = self._validate_text(text)

            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                show_progress_bar=show_progress
            )

            # Validate output
            return self._validate_embedding(embedding, text[:50])

        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            raise

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar (default: True)

        Returns:
            List of embeddings
        """
        try:
            # Handle empty list
            if not texts:
                logger.debug("Empty text list provided, returning empty list")
                return []

            # Validate all texts
            validated_texts = [self._validate_text(text) for text in texts]

            # Generate embeddings
            embeddings = self.model.encode(
                validated_texts,
                convert_to_tensor=False,
                show_progress_bar=show_progress
            )

            # Validate and convert all embeddings
            result = []
            for i, emb in enumerate(embeddings):
                snippet = validated_texts[i][:50] if i < len(validated_texts) else ""
                validated_emb = self._validate_embedding(emb, snippet)
                result.append(validated_emb)

            logger.debug(f"Generated {len(result)} embeddings")
            return result

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise


# Singleton instance
embedding_service = EmbeddingService()
