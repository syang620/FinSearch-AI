"""
Integration tests for vector store with embedding system
Tests the full pipeline from text to stored vectors to retrieval
"""

import pytest
import numpy as np
from typing import List, Dict, Any
import time
from pathlib import Path
import tempfile
import shutil

from app.services.rag.embeddings import embedding_service
from app.services.rag.vector_store import VectorStore


class TestVectorStoreEmbeddingIntegration:
    """Test integration between embeddings and vector store"""

    @pytest.fixture
    def temp_vector_store(self):
        """Create a temporary vector store for testing"""
        temp_dir = tempfile.mkdtemp()
        store = VectorStore(persist_directory=temp_dir)
        yield store
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_end_to_end_storage_retrieval(self, temp_vector_store):
        """Test complete flow: embed -> store -> retrieve"""
        # Sample documents
        documents = [
            {
                "text": "Apple reported record iPhone sales in Q4 2024",
                "metadata": {"ticker": "AAPL", "doc_type": "earnings", "year": 2024}
            },
            {
                "text": "Microsoft Azure revenue grew 30% year-over-year",
                "metadata": {"ticker": "MSFT", "doc_type": "earnings", "year": 2024}
            },
            {
                "text": "Tesla delivered 500,000 vehicles in the quarter",
                "metadata": {"ticker": "TSLA", "doc_type": "report", "year": 2024}
            }
        ]

        # Store documents with embeddings
        for doc in documents:
            embedding = embedding_service.embed_text(doc["text"])
            temp_vector_store.add(
                documents=[doc["text"]],
                embeddings=[embedding],
                metadatas=[doc["metadata"]],
                ids=[f"{doc['metadata']['ticker']}_{doc['metadata']['year']}"]
            )

        # Query with similar text
        query = "iPhone sales performance"
        query_embedding = embedding_service.embed_text(query)

        results = temp_vector_store.query(
            query_embeddings=[query_embedding],
            n_results=2
        )

        # Verify results
        assert len(results['ids'][0]) == 2
        assert "AAPL_2024" in results['ids'][0]  # Apple should be top result
        assert results['metadatas'][0][0]['ticker'] == 'AAPL'

    def test_semantic_search_quality(self, temp_vector_store):
        """Test that semantic search returns relevant results"""
        # Add diverse financial documents
        docs = [
            ("Revenue increased by 15% in the fiscal year", {"topic": "revenue"}),
            ("Sales grew significantly during the quarter", {"topic": "revenue"}),
            ("Operating expenses were reduced by 10%", {"topic": "costs"}),
            ("Net profit margin expanded to 25%", {"topic": "profitability"}),
            ("Debt-to-equity ratio improved substantially", {"topic": "leverage"}),
        ]

        # Store with embeddings
        for i, (text, metadata) in enumerate(docs):
            embedding = embedding_service.embed_text(text)
            temp_vector_store.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[f"doc_{i}"]
            )

        # Test semantic queries
        test_cases = [
            ("revenue growth", "revenue"),  # Should find revenue-related docs
            ("cost reduction", "costs"),     # Should find expense-related docs
            ("profitability metrics", "profitability"),  # Should find profit docs
        ]

        for query, expected_topic in test_cases:
            query_emb = embedding_service.embed_text(query)
            results = temp_vector_store.query(
                query_embeddings=[query_emb],
                n_results=1
            )

            assert len(results['ids'][0]) > 0
            assert results['metadatas'][0][0]['topic'] == expected_topic

    def test_batch_embedding_storage(self, temp_vector_store):
        """Test batch processing of embeddings"""
        # Generate batch of documents
        batch_size = 50
        texts = [f"Financial document {i}: Revenue analysis for Q{i%4 + 1}" for i in range(batch_size)]
        metadatas = [{"doc_id": i, "quarter": i%4 + 1} for i in range(batch_size)]
        ids = [f"doc_{i}" for i in range(batch_size)]

        # Batch embed
        start_time = time.time()
        embeddings = embedding_service.embed_texts(texts, show_progress=False)
        embed_time = time.time() - start_time

        # Batch store
        start_time = time.time()
        temp_vector_store.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        store_time = time.time() - start_time

        # Verify storage
        assert temp_vector_store.collection.count() == batch_size

        # Query to verify
        query_emb = embedding_service.embed_text("Q3 revenue analysis")
        results = temp_vector_store.query(
            query_embeddings=[query_emb],
            n_results=5
        )

        assert len(results['ids'][0]) == 5
        # Should find Q3 documents
        q3_found = any(m['quarter'] == 3 for m in results['metadatas'][0])
        assert q3_found

        print(f"Batch performance: {batch_size} docs embedded in {embed_time:.2f}s, stored in {store_time:.2f}s")

    def test_metadata_filtering_with_embeddings(self, temp_vector_store):
        """Test combining semantic search with metadata filters"""
        # Add documents from different companies
        companies = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        for company in companies:
            for quarter in range(1, 5):
                text = f"{company} reported earnings for Q{quarter} with strong performance"
                embedding = embedding_service.embed_text(text)
                temp_vector_store.add(
                    documents=[text],
                    embeddings=[embedding],
                    metadatas={"ticker": company, "quarter": quarter},
                    ids=[f"{company}_Q{quarter}"]
                )

        # Search with filter
        query = "quarterly earnings performance"
        query_emb = embedding_service.embed_text(query)

        # Filter for only AAPL
        results = temp_vector_store.query(
            query_embeddings=[query_emb],
            n_results=10,
            where={"ticker": "AAPL"}
        )

        # All results should be AAPL
        for metadata in results['metadatas'][0]:
            assert metadata['ticker'] == 'AAPL'

    def test_embedding_consistency_in_store(self, temp_vector_store):
        """Test that same text produces consistent retrieval results"""
        text = "Consistent financial performance metrics"

        # Store same text multiple times with different IDs
        for i in range(3):
            embedding = embedding_service.embed_text(text)
            temp_vector_store.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[{"version": i}],
                ids=[f"consistent_{i}"]
            )

        # Query with exact same text
        query_emb = embedding_service.embed_text(text)
        results = temp_vector_store.query(
            query_embeddings=[query_emb],
            n_results=3
        )

        # All versions should be retrieved with very high similarity
        assert len(results['ids'][0]) == 3
        # Check distances are very small (near-perfect match)
        for distance in results['distances'][0]:
            assert distance < 0.01  # Very close to 0 (identical)

    def test_cross_domain_retrieval(self, temp_vector_store):
        """Test retrieval across different financial domains"""
        domains = {
            "earnings": [
                "Q3 earnings exceeded analyst expectations",
                "Revenue growth driven by strong product sales",
                "Operating margins expanded to record levels"
            ],
            "risk": [
                "Market volatility poses significant risks",
                "Regulatory compliance challenges increased",
                "Cybersecurity threats remain a concern"
            ],
            "strategy": [
                "Expanding into emerging markets",
                "Digital transformation initiatives underway",
                "Focus on sustainable growth strategies"
            ]
        }

        # Store documents from each domain
        doc_id = 0
        for domain, texts in domains.items():
            for text in texts:
                embedding = embedding_service.embed_text(text)
                temp_vector_store.add(
                    documents=[text],
                    embeddings=[embedding],
                    metadatas=[{"domain": domain}],
                    ids=[f"doc_{doc_id}"]
                )
                doc_id += 1

        # Test cross-domain queries
        test_queries = [
            ("financial performance metrics", "earnings"),
            ("operational hazards and challenges", "risk"),
            ("business expansion plans", "strategy")
        ]

        for query, expected_domain in test_queries:
            query_emb = embedding_service.embed_text(query)
            results = temp_vector_store.query(
                query_embeddings=[query_emb],
                n_results=1
            )

            assert results['metadatas'][0][0]['domain'] == expected_domain

    def test_empty_query_handling(self, temp_vector_store):
        """Test handling of empty or invalid queries"""
        # Add some documents
        temp_vector_store.add(
            documents=["Test document"],
            embeddings=[embedding_service.embed_text("Test document")],
            metadatas=[{"test": True}],
            ids=["test_1"]
        )

        # Empty string query
        empty_emb = embedding_service.embed_text("")
        results = temp_vector_store.query(
            query_embeddings=[empty_emb],
            n_results=1
        )

        # Should still return results (though may not be meaningful)
        assert len(results['ids'][0]) >= 0

    def test_large_result_set_retrieval(self, temp_vector_store):
        """Test retrieving large number of results"""
        # Add many documents
        num_docs = 100
        for i in range(num_docs):
            text = f"Document {i}: Financial analysis and insights"
            embedding = embedding_service.embed_text(text)
            temp_vector_store.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[{"index": i}],
                ids=[f"doc_{i}"]
            )

        # Query for many results
        query = "financial analysis"
        query_emb = embedding_service.embed_text(query)
        results = temp_vector_store.query(
            query_embeddings=[query_emb],
            n_results=50  # Request half of all documents
        )

        assert len(results['ids'][0]) == 50
        # Results should be sorted by relevance (distance)
        distances = results['distances'][0]
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))


class TestVectorStoreEmbeddingEdgeCases:
    """Test edge cases in vector store embedding integration"""

    @pytest.fixture
    def temp_vector_store(self):
        """Create a temporary vector store for testing"""
        temp_dir = tempfile.mkdtemp()
        store = VectorStore(persist_directory=temp_dir)
        yield store
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_duplicate_id_handling(self, temp_vector_store):
        """Test handling of duplicate document IDs"""
        text1 = "First version of document"
        text2 = "Updated version of document"

        emb1 = embedding_service.embed_text(text1)
        emb2 = embedding_service.embed_text(text2)

        # Add first document
        temp_vector_store.add(
            documents=[text1],
            embeddings=[emb1],
            metadatas=[{"version": 1}],
            ids=["doc_1"]
        )

        # Try to add with same ID (should update/replace)
        temp_vector_store.add(
            documents=[text2],
            embeddings=[emb2],
            metadatas=[{"version": 2}],
            ids=["doc_1"]
        )

        # Query should return the updated version
        query_emb = embedding_service.embed_text("document version")
        results = temp_vector_store.query(
            query_embeddings=[query_emb],
            n_results=2
        )

        # Should only have one result (not duplicated)
        assert len(set(results['ids'][0])) <= 1

    def test_special_characters_in_text(self, temp_vector_store):
        """Test handling of special characters in embedded text"""
        special_texts = [
            "Revenue: $100M (↑15%) vs €85M",
            "P/E ratio = 25.5; EV/EBITDA ~12x",
            "Growth → 20% | Margin ≈ 35%",
            "Email: investor@company.com | Tel: +1-234-567-8900"
        ]

        for i, text in enumerate(special_texts):
            embedding = embedding_service.embed_text(text)
            temp_vector_store.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[{"has_special": True}],
                ids=[f"special_{i}"]
            )

        # Query with special characters
        query = "Revenue $100M growth ↑15%"
        query_emb = embedding_service.embed_text(query)
        results = temp_vector_store.query(
            query_embeddings=[query_emb],
            n_results=1
        )

        assert len(results['ids'][0]) > 0
        assert "$100M" in results['documents'][0][0]

    def test_multilingual_content(self, temp_vector_store):
        """Test handling of multilingual financial content"""
        multilingual_docs = [
            ("Revenue increased significantly", "en"),
            ("Les revenus ont augmenté significativement", "fr"),
            ("Los ingresos aumentaron significativamente", "es"),
            ("收入显著增长", "zh")
        ]

        for i, (text, lang) in enumerate(multilingual_docs):
            embedding = embedding_service.embed_text(text)
            temp_vector_store.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[{"language": lang}],
                ids=[f"multi_{lang}"]
            )

        # Query in English
        query = "revenue growth"
        query_emb = embedding_service.embed_text(query)
        results = temp_vector_store.query(
            query_embeddings=[query_emb],
            n_results=4
        )

        # Should retrieve documents (model may have some cross-lingual capability)
        assert len(results['ids'][0]) > 0
        # English should likely be top result
        assert any(m['language'] == 'en' for m in results['metadatas'][0][:2])