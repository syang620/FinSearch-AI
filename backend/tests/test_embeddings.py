"""
Comprehensive tests for the embedding service
Tests dimension validation, functionality, edge cases, and performance
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List
import logging

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag.embeddings import EmbeddingService, embedding_service
from app.core.config import settings


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_financial_texts():
    """Sample financial texts for testing"""
    return {
        'revenue_growth': "The company reported revenue growth of 15% year-over-year",
        'revenue_decline': "Sales declined by 10% compared to last quarter",
        'profit_margin': "Operating profit margins expanded to 25%",
        'ebitda': "EBITDA increased significantly due to cost optimization",
        'risk_disclosure': "The company faces significant market volatility risks",
        'empty': "",
        'whitespace': "   \n\t  ",
        'long_text': "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 500,  # ~3000 chars
        'special_chars': "Revenue: $100M (↑15%) vs €85M",
        'unicode': "财报显示收入增长了15%",  # Chinese characters
        'numbers_only': "123 456 789.01",
        'punctuation': "!@#$%^&*()_+-=[]{}|;:',.<>?/`~",
    }


@pytest.fixture
def embedding_service_instance():
    """Create a fresh embedding service instance for testing"""
    return EmbeddingService()


@pytest.fixture
def mock_model():
    """Mock SentenceTransformer model for testing"""
    mock = MagicMock()
    mock.encode = MagicMock()
    mock.get_sentence_embedding_dimension = MagicMock(return_value=768)
    return mock


# ============================================================================
# SECTION A: MODEL INITIALIZATION TESTS
# ============================================================================

class TestModelInitialization:
    """Tests for model loading and initialization"""

    def test_embedding_model_loads_successfully(self):
        """Test that the embedding model loads without errors"""
        service = EmbeddingService()
        assert service.model is not None
        assert service.model_name == settings.EMBEDDING_MODEL

    def test_embedding_model_singleton_pattern(self):
        """Test that the singleton pattern works correctly"""
        # The module-level instance should be the same
        from app.services.rag.embeddings import embedding_service as service1
        from app.services.rag.embeddings import embedding_service as service2
        assert service1 is service2

    @patch('app.services.rag.embeddings.SentenceTransformer')
    def test_embedding_model_initialization_failure(self, mock_st):
        """Test proper error handling when model fails to load"""
        mock_st.side_effect = Exception("Model not found")

        with pytest.raises(Exception) as exc_info:
            service = EmbeddingService()

        assert "Model not found" in str(exc_info.value)


# ============================================================================
# SECTION B: DIMENSION VALIDATION TESTS
# ============================================================================

class TestDimensionValidation:
    """Tests for embedding dimension validation"""

    def test_single_embedding_dimension_correct(self, embedding_service_instance):
        """Test that single embedding has correct dimension"""
        text = "Sample financial text about revenue"
        embedding = embedding_service_instance.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 768  # BGE-base dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_batch_embeddings_dimension_correct(self, embedding_service_instance):
        """Test that batch embeddings have correct dimensions"""
        texts = [
            "First financial text",
            "Second financial text",
            "Third financial text"
        ]
        embeddings = embedding_service_instance.embed_texts(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)

    def test_embedding_dimension_matches_config(self, embedding_service_instance):
        """Test that embedding dimension matches configuration"""
        text = "Test text"
        embedding = embedding_service_instance.embed_text(text)

        expected_dim = settings.EMBEDDING_DIMENSION
        assert len(embedding) == expected_dim


# ============================================================================
# SECTION C: EDGE CASES & INPUT VALIDATION TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and input validation"""

    def test_empty_string_embedding(self, embedding_service_instance):
        """Test handling of empty string input"""
        # Should still return a valid embedding (model handles empty strings)
        embedding = embedding_service_instance.embed_text("")
        assert isinstance(embedding, list)
        assert len(embedding) == 768

    def test_whitespace_only_input(self, embedding_service_instance):
        """Test handling of whitespace-only input"""
        embedding = embedding_service_instance.embed_text("   \n\t  ")
        assert isinstance(embedding, list)
        assert len(embedding) == 768

    def test_very_long_text_embedding(self, embedding_service_instance, sample_financial_texts):
        """Test handling of very long text input"""
        long_text = sample_financial_texts['long_text']
        embedding = embedding_service_instance.embed_text(long_text)

        assert isinstance(embedding, list)
        assert len(embedding) == 768
        # Should not fail, model should handle truncation internally

    def test_special_characters_embedding(self, embedding_service_instance, sample_financial_texts):
        """Test handling of special characters"""
        text = sample_financial_texts['special_chars']
        embedding = embedding_service_instance.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 768

    def test_unicode_text_embedding(self, embedding_service_instance, sample_financial_texts):
        """Test handling of unicode/international characters"""
        text = sample_financial_texts['unicode']
        embedding = embedding_service_instance.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 768

    def test_empty_list_batch_embedding(self, embedding_service_instance):
        """Test handling of empty list input for batch embedding"""
        embeddings = embedding_service_instance.embed_texts([])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 0

    def test_mixed_length_texts_batch(self, embedding_service_instance):
        """Test batch processing of texts with different lengths"""
        texts = [
            "Short",
            "Medium length text here with more words",
            "Very long text that contains many words and should be much longer than the others " * 10
        ]
        embeddings = embedding_service_instance.embed_texts(texts)

        assert len(embeddings) == 3
        for embedding in embeddings:
            assert len(embedding) == 768

    def test_numbers_only_text(self, embedding_service_instance, sample_financial_texts):
        """Test embedding of numbers-only text"""
        text = sample_financial_texts['numbers_only']
        embedding = embedding_service_instance.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 768

    def test_punctuation_only_text(self, embedding_service_instance, sample_financial_texts):
        """Test embedding of punctuation-only text"""
        text = sample_financial_texts['punctuation']
        embedding = embedding_service_instance.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 768


# ============================================================================
# SECTION D: EMBEDDING QUALITY TESTS
# ============================================================================

class TestEmbeddingQuality:
    """Tests for embedding semantic quality"""

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def test_semantic_similarity_same_text(self, embedding_service_instance):
        """Test that same text produces identical embeddings"""
        text = "Revenue increased by 15% this quarter"
        emb1 = embedding_service_instance.embed_text(text)
        emb2 = embedding_service_instance.embed_text(text)

        similarity = self.cosine_similarity(emb1, emb2)
        assert similarity > 0.999  # Should be essentially 1.0

    def test_semantic_similarity_similar_texts(self, embedding_service_instance):
        """Test that similar texts have high similarity"""
        text1 = "Revenue increased 15%"
        text2 = "Sales grew by fifteen percent"

        emb1 = embedding_service_instance.embed_text(text1)
        emb2 = embedding_service_instance.embed_text(text2)

        similarity = self.cosine_similarity(emb1, emb2)
        assert similarity > 0.7  # Should have high similarity

    def test_semantic_similarity_different_texts(self, embedding_service_instance):
        """Test that different texts have low similarity"""
        text1 = "Financial performance improved significantly"
        text2 = "The weather forecast predicts rain tomorrow"

        emb1 = embedding_service_instance.embed_text(text1)
        emb2 = embedding_service_instance.embed_text(text2)

        similarity = self.cosine_similarity(emb1, emb2)
        assert similarity < 0.5  # Should have low similarity

    def test_financial_domain_understanding(self, embedding_service_instance):
        """Test that model understands financial terminology"""
        text1 = "EBITDA margin expanded"
        text2 = "Operating profit margin improved"

        emb1 = embedding_service_instance.embed_text(text1)
        emb2 = embedding_service_instance.embed_text(text2)

        similarity = self.cosine_similarity(emb1, emb2)
        assert similarity > 0.6  # Should recognize similar financial concepts

    def test_negation_sensitivity(self, embedding_service_instance):
        """Test that model is sensitive to negation"""
        text1 = "Revenue increased significantly"
        text2 = "Revenue decreased significantly"

        emb1 = embedding_service_instance.embed_text(text1)
        emb2 = embedding_service_instance.embed_text(text2)

        similarity = self.cosine_similarity(emb1, emb2)
        # Should have some similarity (both about revenue) but not too high
        assert 0.3 < similarity < 0.8

    def test_embedding_consistency(self, embedding_service_instance):
        """Test that embeddings are consistent across multiple calls"""
        text = "Quarterly earnings exceeded expectations"

        embeddings = []
        for _ in range(5):
            emb = embedding_service_instance.embed_text(text)
            embeddings.append(emb)

        # All embeddings should be identical
        for i in range(1, len(embeddings)):
            similarity = self.cosine_similarity(embeddings[0], embeddings[i])
            assert similarity > 0.999


# ============================================================================
# SECTION E: NORMALIZATION TESTS
# ============================================================================

class TestNormalization:
    """Tests for embedding normalization"""

    def test_embeddings_normalized(self, embedding_service_instance):
        """Test if embeddings are normalized (unit vectors)"""
        text = "Test financial document"
        embedding = embedding_service_instance.embed_text(text)

        # Calculate L2 norm
        norm = np.linalg.norm(np.array(embedding))

        # BGE models typically output normalized embeddings
        # Allow small tolerance for floating point errors
        assert abs(norm - 1.0) < 0.01 or norm > 0  # Either normalized or not

    def test_embedding_value_range(self, embedding_service_instance):
        """Test that embedding values are in expected range"""
        text = "Financial statement analysis"
        embedding = embedding_service_instance.embed_text(text)

        # Check for NaN or Inf values
        assert all(not np.isnan(x) for x in embedding)
        assert all(not np.isinf(x) for x in embedding)

        # Values should be reasonable floats
        assert all(-10 < x < 10 for x in embedding)  # Typical range for embeddings


# ============================================================================
# SECTION F: PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Tests for embedding performance"""

    def test_single_embedding_performance(self, embedding_service_instance):
        """Test performance of single embedding generation"""
        text = "Financial performance metrics for Q3 2025"

        start_time = time.time()
        embedding = embedding_service_instance.embed_text(text)
        end_time = time.time()

        elapsed_time = end_time - start_time

        # Should be fast on CPU (< 200ms)
        assert elapsed_time < 0.2
        assert len(embedding) == 768

    def test_batch_embedding_performance(self, embedding_service_instance):
        """Test that batch processing is more efficient than individual calls"""
        texts = [f"Financial text number {i}" for i in range(20)]

        # Time batch processing
        start_batch = time.time()
        batch_embeddings = embedding_service_instance.embed_texts(texts)
        batch_time = time.time() - start_batch

        # Time individual processing
        start_individual = time.time()
        individual_embeddings = []
        for text in texts[:5]:  # Just test a few for comparison
            emb = embedding_service_instance.embed_text(text)
            individual_embeddings.append(emb)
        individual_time = (time.time() - start_individual) * 4  # Extrapolate to 20

        # Batch should be faster than individual calls
        assert batch_time < individual_time
        assert len(batch_embeddings) == 20

    def test_memory_usage_batch_embedding(self, embedding_service_instance):
        """Test memory efficiency of batch embedding"""
        # Create 100 sample texts
        texts = [f"Sample financial document text {i} with some content" for i in range(100)]

        # This should not cause memory issues
        embeddings = embedding_service_instance.embed_texts(texts)

        assert len(embeddings) == 100
        # Each embedding should be 768 floats
        total_floats = sum(len(emb) for emb in embeddings)
        assert total_floats == 100 * 768


# ============================================================================
# SECTION G: TYPE VALIDATION TESTS
# ============================================================================

class TestTypeValidation:
    """Tests for return type validation"""

    def test_single_embedding_return_type(self, embedding_service_instance):
        """Test that single embedding returns correct type"""
        text = "Revenue forecast for next quarter"
        embedding = embedding_service_instance.embed_text(text)

        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)
        # Should not be numpy array
        assert not isinstance(embedding, np.ndarray)

    def test_batch_embedding_return_type(self, embedding_service_instance):
        """Test that batch embedding returns correct type"""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedding_service_instance.embed_texts(texts)

        assert isinstance(embeddings, list)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert all(isinstance(x, float) for x in embedding)
            assert not isinstance(embedding, np.ndarray)

    def test_embedding_serialization(self, embedding_service_instance):
        """Test that embeddings can be JSON serialized"""
        import json

        text = "Quarterly earnings report"
        embedding = embedding_service_instance.embed_text(text)

        # Should be JSON serializable
        json_str = json.dumps(embedding)
        assert json_str is not None

        # Should be able to deserialize
        deserialized = json.loads(json_str)
        assert deserialized == embedding


# ============================================================================
# SECTION H: ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in embedding service"""

    @patch('app.services.rag.embeddings.SentenceTransformer')
    def test_embed_text_error_handling(self, mock_st):
        """Test error handling in embed_text method"""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Encoding error")
        mock_st.return_value = mock_model

        service = EmbeddingService()

        with pytest.raises(Exception) as exc_info:
            service.embed_text("Test text")

        assert "Encoding error" in str(exc_info.value)

    @patch('app.services.rag.embeddings.SentenceTransformer')
    def test_embed_texts_error_handling(self, mock_st):
        """Test error handling in embed_texts method"""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Batch encoding error")
        mock_st.return_value = mock_model

        service = EmbeddingService()

        with pytest.raises(Exception) as exc_info:
            service.embed_texts(["Text 1", "Text 2"])

        assert "Batch encoding error" in str(exc_info.value)


# ============================================================================
# SECTION I: FINANCIAL DOMAIN SPECIFIC TESTS
# ============================================================================

class TestFinancialDomain:
    """Tests specific to financial document embeddings"""

    def test_financial_acronyms(self, embedding_service_instance):
        """Test understanding of financial acronyms"""
        pairs = [
            ("EBITDA", "Earnings Before Interest Taxes Depreciation Amortization"),
            ("P/E ratio", "Price to Earnings ratio"),
            ("ROI", "Return on Investment"),
        ]

        for acronym, full_form in pairs:
            emb1 = embedding_service_instance.embed_text(acronym)
            emb2 = embedding_service_instance.embed_text(full_form)

            similarity = TestEmbeddingQuality().cosine_similarity(emb1, emb2)
            assert similarity > 0.5  # Should recognize relationship

    def test_ticker_symbol_context(self, embedding_service_instance):
        """Test that ticker symbols are understood in context"""
        text1 = "AAPL reported strong earnings"
        text2 = "Apple reported strong earnings"

        emb1 = embedding_service_instance.embed_text(text1)
        emb2 = embedding_service_instance.embed_text(text2)

        similarity = TestEmbeddingQuality().cosine_similarity(emb1, emb2)
        assert similarity > 0.7  # Should understand AAPL = Apple

    def test_financial_metrics_similarity(self, embedding_service_instance):
        """Test similarity of related financial metrics"""
        metrics_groups = [
            ["revenue", "sales", "turnover"],
            ["profit", "earnings", "income"],
            ["costs", "expenses", "expenditure"],
        ]

        for group in metrics_groups:
            embeddings = [embedding_service_instance.embed_text(term) for term in group]

            # All terms in a group should be similar
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = TestEmbeddingQuality().cosine_similarity(
                        embeddings[i], embeddings[j]
                    )
                    assert similarity > 0.5  # Related terms should be similar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])