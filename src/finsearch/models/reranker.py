"""
Document reranking module for RAG pipeline.
Consolidates multiple reranker variants into one optimized implementation.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RankedDocument:
    """Document with reranking score"""
    text: str
    metadata: Dict[str, Any]
    retrieval_score: float
    rerank_score: float
    doc_id: str

    @property
    def combined_score(self) -> float:
        """Combined score for final ranking"""
        # You can adjust this formula based on experiments
        return 0.3 * self.retrieval_score + 0.7 * self.rerank_score


class Reranker:
    """
    Optimized cross-encoder reranker for document ranking.
    Consolidates standard, batch, hybrid, and parallel reranker implementations.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
        use_cache: bool = True
    ):
        """
        Initialize the reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device to run on ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            use_cache: Whether to cache reranking scores
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_cache = use_cache

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model and tokenizer
        self._load_model()

        # Score cache for efficiency
        self.score_cache = {} if use_cache else None

    def _load_model(self):
        """Load the cross-encoder model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded reranker model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None,
        return_scores: bool = False
    ) -> List[RankedDocument]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Search query
            documents: List of documents to rerank (can be Document objects or dicts)
            top_k: Return only top-k documents (None for all)
            return_scores: Whether to return raw scores

        Returns:
            List of RankedDocument objects sorted by relevance
        """
        if not documents:
            return []

        # Convert documents to standardized format
        doc_texts, doc_metadata, doc_ids, retrieval_scores = self._prepare_documents(documents)

        # Get reranking scores
        scores = self._compute_scores_batch(query, doc_texts)

        # Create ranked documents
        ranked_docs = []
        for i, (text, metadata, doc_id, ret_score, rerank_score) in enumerate(
            zip(doc_texts, doc_metadata, doc_ids, retrieval_scores, scores)
        ):
            ranked_doc = RankedDocument(
                text=text,
                metadata=metadata,
                retrieval_score=ret_score,
                rerank_score=float(rerank_score),
                doc_id=doc_id
            )
            ranked_docs.append(ranked_doc)

        # Sort by reranking score (or combined score)
        ranked_docs.sort(key=lambda x: x.rerank_score, reverse=True)

        # Return top-k if specified
        if top_k is not None:
            ranked_docs = ranked_docs[:top_k]

        if return_scores:
            return ranked_docs, scores
        return ranked_docs

    def _prepare_documents(self, documents: List[Any]) -> Tuple[List[str], List[Dict], List[str], List[float]]:
        """
        Prepare documents for reranking.
        Handles different document formats.
        """
        texts = []
        metadatas = []
        ids = []
        retrieval_scores = []

        for i, doc in enumerate(documents):
            if hasattr(doc, 'text'):
                # Document object
                texts.append(doc.text)
                metadatas.append(getattr(doc, 'metadata', {}))
                ids.append(getattr(doc, 'doc_id', f"doc_{i}"))
                retrieval_scores.append(getattr(doc, 'score', 0.0))
            elif isinstance(doc, dict):
                # Dictionary format
                texts.append(doc.get('text', doc.get('content', '')))
                metadatas.append(doc.get('metadata', {}))
                ids.append(doc.get('id', f"doc_{i}"))
                retrieval_scores.append(doc.get('score', 0.0))
            else:
                # Plain text
                texts.append(str(doc))
                metadatas.append({})
                ids.append(f"doc_{i}")
                retrieval_scores.append(0.0)

        return texts, metadatas, ids, retrieval_scores

    def _compute_scores_batch(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Compute reranking scores in batches for efficiency.
        This replaces the batch and parallel reranker functionality.
        """
        all_scores = []

        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]

            # Check cache
            batch_scores = []
            uncached_docs = []
            uncached_indices = []

            if self.use_cache and self.score_cache is not None:
                for j, doc in enumerate(batch_docs):
                    cache_key = f"{query}||{doc[:200]}"  # Use prefix for cache key
                    if cache_key in self.score_cache:
                        batch_scores.append(self.score_cache[cache_key])
                    else:
                        uncached_docs.append(doc)
                        uncached_indices.append(j)
            else:
                uncached_docs = batch_docs
                uncached_indices = list(range(len(batch_docs)))

            # Compute scores for uncached documents
            if uncached_docs:
                # Prepare inputs
                inputs = self.tokenizer(
                    [query] * len(uncached_docs),
                    uncached_docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get scores
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                    # Apply sigmoid for probability scores
                    if logits.shape[1] == 1:
                        # Single score output
                        scores = torch.sigmoid(logits).squeeze().cpu().numpy()
                    else:
                        # Multi-class output (take positive class)
                        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                # Handle single document case
                if isinstance(scores, (float, np.float32, np.float64)):
                    scores = [scores]

                # Update cache
                if self.use_cache and self.score_cache is not None:
                    for doc, score in zip(uncached_docs, scores):
                        cache_key = f"{query}||{doc[:200]}"
                        self.score_cache[cache_key] = float(score)

                # Insert scores in correct positions
                if self.use_cache:
                    final_scores = [None] * len(batch_docs)
                    for idx, score in zip(uncached_indices, scores):
                        final_scores[idx] = float(score)
                    for idx, score in enumerate(batch_scores):
                        if final_scores[idx] is None:
                            final_scores[idx] = score
                    all_scores.extend(final_scores)
                else:
                    all_scores.extend(scores)
            else:
                # All cached
                all_scores.extend(batch_scores)

        return np.array(all_scores)

    def rerank_diverse(
        self,
        query: str,
        documents: List[Any],
        top_k: int = 5,
        diversity_weight: float = 0.3
    ) -> List[RankedDocument]:
        """
        Rerank with diversity to reduce redundancy.
        This provides some of the hybrid reranker functionality.
        """
        # Get initial reranking
        ranked_docs = self.rerank(query, documents)

        if len(ranked_docs) <= top_k or diversity_weight == 0:
            return ranked_docs[:top_k]

        # MMR-style diverse reranking
        selected = []
        remaining = ranked_docs.copy()

        # Select first document (highest relevance)
        selected.append(remaining.pop(0))

        # Select remaining documents considering diversity
        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_idx = -1

            for i, doc in enumerate(remaining):
                # Relevance score
                rel_score = doc.rerank_score

                # Diversity score (negative of max similarity to selected)
                div_score = 1.0
                for sel_doc in selected:
                    # Simple text overlap as similarity proxy
                    overlap = len(set(doc.text.split()) & set(sel_doc.text.split()))
                    similarity = overlap / max(len(doc.text.split()), 1)
                    div_score = min(div_score, 1.0 - similarity)

                # Combined score
                combined = (1 - diversity_weight) * rel_score + diversity_weight * div_score

                if combined > best_score:
                    best_score = combined
                    best_idx = i

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))

        return selected

    def clear_cache(self):
        """Clear the score cache"""
        if self.score_cache is not None:
            self.score_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics"""
        stats = {
            'model_name': self.model_name,
            'device': str(self.device),
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'cache_enabled': self.use_cache,
            'cache_size': len(self.score_cache) if self.score_cache else 0
        }
        return stats