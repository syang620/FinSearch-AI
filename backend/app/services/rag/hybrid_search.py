"""
Hybrid Search Service combining BM25 (lexical) and Dense (semantic) retrieval
with Reciprocal Rank Fusion (RRF) for superior financial document search
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from app.services.rag.vector_store import vector_store
from app.services.rag.bm25_index import bm25_index

logger = logging.getLogger(__name__)


class HybridSearchService:
    """
    Combines BM25 keyword search with dense vector semantic search
    using Reciprocal Rank Fusion for optimal retrieval quality
    """

    def __init__(self, rrf_k: int = 60):
        """
        Initialize hybrid search with RRF parameter

        Args:
            rrf_k: RRF constant (typically 60, lower = more weight to top ranks)
        """
        self.rrf_k = rrf_k
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        logger.info(f"Hybrid search initialized with RRF k={rrf_k}")

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        debug: bool = False
    ) -> Dict[str, List]:
        """
        Perform hybrid search combining BM25 and dense retrieval

        Args:
            query: Search query text
            n_results: Number of results to return
            where: Optional metadata filters
            bm25_weight: Weight for BM25 scores (0-1)
            dense_weight: Weight for dense scores (0-1)
            debug: If True, include debug info in results

        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'scores', and optionally 'debug'
        """
        try:
            # Normalize weights
            total_weight = bm25_weight + dense_weight
            if total_weight > 0:
                bm25_weight = bm25_weight / total_weight
                dense_weight = dense_weight / total_weight
            else:
                bm25_weight = dense_weight = 0.5

            # Fetch more results from each system to ensure good coverage
            fetch_k = min(n_results * 3, 100)

            # Run BM25 search
            bm25_results = self._search_bm25(query, fetch_k, where)

            # Run dense vector search
            dense_results = self._search_dense(query, fetch_k, where)

            # Apply RRF fusion
            fused_results = self._apply_rrf(
                bm25_results,
                dense_results,
                bm25_weight,
                dense_weight,
                n_results,
                debug
            )

            if debug:
                fused_results['debug'] = {
                    'bm25_hits': len(bm25_results['ids'][0]) if bm25_results['ids'] else 0,
                    'dense_hits': len(dense_results['ids'][0]) if dense_results['ids'] else 0,
                    'query': query,
                    'weights': {'bm25': bm25_weight, 'dense': dense_weight}
                }

            return fused_results

        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            # Fallback to dense search only
            return self._search_dense(query, n_results, where)

    def _search_bm25(
        self,
        query: str,
        n_results: int,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List]:
        """Run BM25 keyword search"""
        try:
            results = bm25_index.search(query, n_results, where)
            logger.debug(f"BM25 returned {len(results['ids'][0]) if results['ids'] else 0} results")
            return results
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return {'ids': [[]], 'documents': [[]], 'scores': [[]], 'metadatas': [[]]}

    def _search_dense(
        self,
        query: str,
        n_results: int,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List]:
        """Run dense vector semantic search"""
        try:
            results = vector_store.search(query, n_results, where)
            logger.debug(f"Dense search returned {len(results['ids'][0]) if results['ids'] else 0} results")
            return results
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return {'ids': [[]], 'documents': [[]], 'distances': [[]], 'metadatas': [[]]}

    def _apply_rrf(
        self,
        bm25_results: Dict[str, List],
        dense_results: Dict[str, List],
        bm25_weight: float,
        dense_weight: float,
        n_results: int,
        debug: bool = False
    ) -> Dict[str, List]:
        """
        Apply Reciprocal Rank Fusion to merge BM25 and dense results

        RRF formula: score(d) = Σ w_i / (k + rank_i(d))
        where w_i is the weight for retrieval method i
        """
        # Create mapping of chunk_id to data
        chunk_data = {}
        rrf_scores = defaultdict(float)
        debug_info = defaultdict(dict) if debug else None

        # Process BM25 results
        if bm25_results['ids'] and bm25_results['ids'][0]:
            bm25_ids = bm25_results['ids'][0]
            bm25_docs = bm25_results['documents'][0]
            bm25_metas = bm25_results['metadatas'][0]
            bm25_scores = bm25_results.get('scores', [[]])[0]

            for rank, chunk_id in enumerate(bm25_ids, 1):
                # Store document data
                if chunk_id not in chunk_data:
                    idx = rank - 1
                    chunk_data[chunk_id] = {
                        'document': bm25_docs[idx],
                        'metadata': bm25_metas[idx]
                    }

                # Calculate RRF score
                rrf_score = bm25_weight / (self.rrf_k + rank)
                rrf_scores[chunk_id] += rrf_score

                if debug:
                    debug_info[chunk_id]['bm25_rank'] = rank
                    if bm25_scores and idx < len(bm25_scores):
                        debug_info[chunk_id]['bm25_score'] = bm25_scores[idx]
                    debug_info[chunk_id]['bm25_rrf'] = rrf_score

        # Process dense results
        if dense_results['ids'] and dense_results['ids'][0]:
            dense_ids = dense_results['ids'][0]
            dense_docs = dense_results['documents'][0]
            dense_metas = dense_results['metadatas'][0]
            dense_distances = dense_results.get('distances', [[]])[0]

            for rank, chunk_id in enumerate(dense_ids, 1):
                # Store document data
                if chunk_id not in chunk_data:
                    idx = rank - 1
                    chunk_data[chunk_id] = {
                        'document': dense_docs[idx],
                        'metadata': dense_metas[idx]
                    }

                # Calculate RRF score
                rrf_score = dense_weight / (self.rrf_k + rank)
                rrf_scores[chunk_id] += rrf_score

                if debug:
                    debug_info[chunk_id]['dense_rank'] = rank
                    if dense_distances and idx < len(dense_distances):
                        debug_info[chunk_id]['dense_distance'] = dense_distances[idx]
                    debug_info[chunk_id]['dense_rrf'] = rrf_score

        # Sort by RRF score and get top n_results
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_results]

        # Build final results
        ids = []
        documents = []
        metadatas = []
        scores = []

        for chunk_id, rrf_score in sorted_chunks:
            data = chunk_data[chunk_id]
            ids.append(chunk_id)
            documents.append(data['document'])
            metadatas.append(data['metadata'])
            scores.append(rrf_score)

            if debug and chunk_id in debug_info:
                debug_info[chunk_id]['final_rrf_score'] = rrf_score

        results = {
            'ids': [ids],  # Nested list to match ChromaDB format
            'documents': [documents],
            'metadatas': [metadatas],
            'scores': [scores]
        }

        if debug:
            results['debug_ranking'] = [debug_info[chunk_id] for chunk_id in ids]

        logger.info(
            f"RRF fusion: {len(rrf_scores)} unique chunks → {len(ids)} results "
            f"(BM25: {len([k for k in rrf_scores if 'bm25_rank' in debug_info.get(k, {})])}, "
            f"Dense: {len([k for k in rrf_scores if 'dense_rank' in debug_info.get(k, {})])})"
        )

        return results

    def explain_retrieval(
        self,
        query: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Explain why certain documents were retrieved for a query

        Args:
            query: Search query
            n_results: Number of results to explain

        Returns:
            Detailed explanation of retrieval decisions
        """
        # Run search with debug info
        results = self.search(query, n_results, debug=True)

        explanation = {
            'query': query,
            'top_results': []
        }

        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0][:n_results]):
                doc_snippet = results['documents'][0][i][:200] + '...'
                metadata = results['metadatas'][0][i]

                result_info = {
                    'rank': i + 1,
                    'chunk_id': chunk_id,
                    'ticker': metadata.get('ticker', 'N/A'),
                    'doc_type': metadata.get('doc_type', 'N/A'),
                    'snippet': doc_snippet,
                    'rrf_score': results['scores'][0][i]
                }

                if 'debug_ranking' in results and i < len(results['debug_ranking']):
                    debug = results['debug_ranking'][i]
                    result_info['retrieval_details'] = {
                        'bm25': {
                            'rank': debug.get('bm25_rank', 'Not retrieved'),
                            'contribution': debug.get('bm25_rrf', 0)
                        },
                        'dense': {
                            'rank': debug.get('dense_rank', 'Not retrieved'),
                            'contribution': debug.get('dense_rrf', 0)
                        }
                    }

                explanation['top_results'].append(result_info)

        return explanation


# Singleton instance for easy import
hybrid_search = HybridSearchService()