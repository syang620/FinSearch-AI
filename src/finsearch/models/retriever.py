"""
Unified retrieval module for RAG pipeline.
Consolidates vector_store.py, bm25_index.py, and hybrid_search.py.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions
import pickle
from pathlib import Path


@dataclass
class Document:
    """Represents a retrieved document"""
    text: str
    metadata: Dict[str, Any]
    score: float
    doc_id: str


class HybridRetriever:
    """
    Unified retriever supporting dense, sparse, and hybrid search.
    Consolidates multiple retrieval services into one interface.
    """

    def __init__(
        self,
        collection_name: str = "finsearch",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        use_hybrid: bool = True,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        persist_directory: str = "data/processed/embeddings"
    ):
        """
        Initialize the hybrid retriever.

        Args:
            collection_name: ChromaDB collection name
            embedding_model: HuggingFace embedding model name
            use_hybrid: Whether to use hybrid search (dense + sparse)
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse retrieval scores
            persist_directory: Directory to persist vector store
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.use_hybrid = use_hybrid
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.persist_directory = persist_directory

        # Initialize ChromaDB for dense retrieval
        self._init_vector_store()

        # BM25 components for sparse retrieval
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadata = []

    def _init_vector_store(self):
        """Initialize ChromaDB vector store"""
        # Create embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )

        # Initialize persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to both vector store and BM25 index.

        Args:
            documents: List of document texts
            metadatas: List of metadata dicts
            ids: Optional document IDs
        """
        if not documents:
            return

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        # Add to ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        # Update BM25 index
        self._update_bm25_index(documents, metadatas)

    def _update_bm25_index(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """Update BM25 index with new documents"""
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]

        # Add to existing documents
        self.bm25_documents.extend(documents)
        self.bm25_metadata.extend(metadatas)

        # Rebuild BM25 index
        all_tokenized = [doc.lower().split() for doc in self.bm25_documents]
        self.bm25_index = BM25Okapi(all_tokenized)

    def retrieve(
        self,
        query: str,
        k: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Main retrieval method supporting dense, sparse, or hybrid.

        Args:
            query: Search query
            k: Number of documents to retrieve
            filter_dict: Optional metadata filters

        Returns:
            List of retrieved Document objects
        """
        if self.use_hybrid and self.bm25_index:
            return self._hybrid_retrieve(query, k, filter_dict)
        else:
            return self._dense_retrieve(query, k, filter_dict)

    def _dense_retrieve(
        self,
        query: str,
        k: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Dense retrieval using vector similarity"""
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter_dict
        )

        # Convert to Document objects
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc_text in enumerate(results['documents'][0]):
                doc = Document(
                    text=doc_text,
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                    score=1.0 - results['distances'][0][i] if results['distances'] else 0.0,
                    doc_id=results['ids'][0][i] if results['ids'] else f"doc_{i}"
                )
                documents.append(doc)

        return documents

    def _sparse_retrieve(
        self,
        query: str,
        k: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Sparse retrieval using BM25"""
        if not self.bm25_index:
            return []

        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)

        # Apply filters if provided
        valid_indices = list(range(len(scores)))
        if filter_dict:
            valid_indices = [
                i for i in valid_indices
                if self._match_filter(self.bm25_metadata[i], filter_dict)
            ]

        # Get top-k indices
        valid_scores = [(i, scores[i]) for i in valid_indices]
        valid_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in valid_scores[:k]]

        # Convert to Document objects
        documents = []
        for idx in top_indices:
            doc = Document(
                text=self.bm25_documents[idx],
                metadata=self.bm25_metadata[idx],
                score=scores[idx],
                doc_id=f"bm25_doc_{idx}"
            )
            documents.append(doc)

        return documents

    def _hybrid_retrieve(
        self,
        query: str,
        k: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Hybrid retrieval combining dense and sparse methods.
        Uses reciprocal rank fusion for score combination.
        """
        # Get results from both methods
        dense_docs = self._dense_retrieve(query, k * 2, filter_dict)
        sparse_docs = self._sparse_retrieve(query, k * 2, filter_dict)

        # Reciprocal Rank Fusion
        doc_scores = {}

        # Process dense results
        for rank, doc in enumerate(dense_docs):
            doc_id = doc.text[:100]  # Use text prefix as ID for matching
            score = self.dense_weight / (rank + 1)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id][0] + score, doc)
            else:
                doc_scores[doc_id] = (score, doc)

        # Process sparse results
        for rank, doc in enumerate(sparse_docs):
            doc_id = doc.text[:100]  # Use text prefix as ID for matching
            score = self.sparse_weight / (rank + 1)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id][0] + score, doc)
            else:
                doc_scores[doc_id] = (score, doc)

        # Sort by combined score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)

        # Return top-k documents
        result_docs = []
        for score, doc in sorted_docs[:k]:
            doc.score = score  # Update with combined score
            result_docs.append(doc)

        return result_docs

    def _match_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    def save_bm25_index(self, path: str = "data/processed/indexes/bm25_index.pkl"):
        """Save BM25 index to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'index': self.bm25_index,
                'documents': self.bm25_documents,
                'metadata': self.bm25_metadata
            }, f)

    def load_bm25_index(self, path: str = "data/processed/indexes/bm25_index.pkl"):
        """Load BM25 index from disk"""
        if Path(path).exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.bm25_index = data['index']
                self.bm25_documents = data['documents']
                self.bm25_metadata = data['metadata']
            return True
        return False

    def clear(self):
        """Clear all indices"""
        # Clear ChromaDB collection
        self.chroma_client.delete_collection(self.collection_name)
        self._init_vector_store()

        # Clear BM25
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadata = []

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        stats = {
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model,
            'use_hybrid': self.use_hybrid,
            'dense_weight': self.dense_weight,
            'sparse_weight': self.sparse_weight,
            'vector_store_count': self.collection.count(),
            'bm25_document_count': len(self.bm25_documents)
        }
        return stats