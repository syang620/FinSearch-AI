"""
Evaluation metrics for RAG pipeline.
Includes retrieval and generation metrics.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    query: str
    retrieved: List[str]
    relevant: List[str]
    metrics: Dict[str, float]
    k: int


class RAGEvaluator:
    """
    Evaluator for RAG pipeline performance.
    Supports various retrieval and generation metrics.
    """

    def __init__(
        self,
        retriever=None,
        reranker=None,
        metrics: List[str] = None
    ):
        """
        Initialize the evaluator.

        Args:
            retriever: Retriever instance
            reranker: Optional reranker instance
            metrics: List of metrics to compute
        """
        self.retriever = retriever
        self.reranker = reranker
        self.metrics = metrics or ["precision", "recall", "mrr", "ndcg"]

    def evaluate(
        self,
        queries: List[str],
        relevant_docs: List[List[str]],
        k_values: List[int] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate retrieval performance on a test set.

        Args:
            queries: List of query strings
            relevant_docs: List of relevant document IDs for each query
            k_values: List of k values to evaluate at

        Returns:
            List of evaluation results
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]

        results = []

        for query, relevant in zip(queries, relevant_docs):
            for k in k_values:
                # Retrieve documents
                retrieved = self._retrieve_and_process(query, k)

                # Calculate metrics
                metrics = self._calculate_metrics(retrieved, relevant, k)

                result = EvaluationResult(
                    query=query,
                    retrieved=retrieved[:k],
                    relevant=relevant,
                    metrics=metrics,
                    k=k
                )
                results.append(result)

                logger.debug(f"Query: {query[:50]}... | P@{k}: {metrics.get('precision', 0):.3f}")

        return results

    def _retrieve_and_process(self, query: str, k: int) -> List[str]:
        """Retrieve documents and optionally rerank"""
        if not self.retriever:
            return []

        # Retrieve
        docs = self.retriever.retrieve(query, k=k*2)  # Retrieve more for reranking

        # Rerank if available
        if self.reranker and docs:
            docs = self.reranker.rerank(query, docs, top_k=k)

        # Extract document IDs
        doc_ids = []
        for doc in docs:
            if hasattr(doc, 'doc_id'):
                doc_ids.append(doc.doc_id)
            elif hasattr(doc, 'metadata') and 'doc_id' in doc.metadata:
                doc_ids.append(doc.metadata['doc_id'])
            else:
                # Use text hash as ID
                doc_ids.append(str(hash(doc.text if hasattr(doc, 'text') else str(doc))[:8]))

        return doc_ids

    def _calculate_metrics(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {}

        # Get top-k retrieved
        retrieved_at_k = retrieved[:k]

        # Precision@k
        if "precision" in self.metrics or "precision_at_k" in self.metrics:
            metrics["precision"] = self.precision_at_k(retrieved_at_k, relevant)

        # Recall@k
        if "recall" in self.metrics or "recall_at_k" in self.metrics:
            metrics["recall"] = self.recall_at_k(retrieved_at_k, relevant)

        # F1@k
        if "f1" in self.metrics:
            p = metrics.get("precision", self.precision_at_k(retrieved_at_k, relevant))
            r = metrics.get("recall", self.recall_at_k(retrieved_at_k, relevant))
            metrics["f1"] = self.f1_score(p, r)

        # MRR (Mean Reciprocal Rank)
        if "mrr" in self.metrics:
            metrics["mrr"] = self.mean_reciprocal_rank(retrieved_at_k, relevant)

        # NDCG (Normalized Discounted Cumulative Gain)
        if "ndcg" in self.metrics:
            metrics["ndcg"] = self.ndcg_at_k(retrieved_at_k, relevant, k)

        # MAP (Mean Average Precision)
        if "map" in self.metrics:
            metrics["map"] = self.average_precision(retrieved_at_k, relevant)

        return metrics

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate precision@k"""
        if not retrieved:
            return 0.0

        relevant_set = set(relevant)
        hits = sum(1 for doc_id in retrieved if doc_id in relevant_set)
        return hits / len(retrieved)

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate recall@k"""
        if not relevant:
            return 0.0

        relevant_set = set(relevant)
        hits = sum(1 for doc_id in retrieved if doc_id in relevant_set)
        return hits / len(relevant)

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        relevant_set = set(relevant)

        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate NDCG@k"""
        relevant_set = set(relevant)

        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant_set:
                # Binary relevance: 1 if relevant, 0 otherwise
                dcg += 1.0 / np.log2(i + 2)  # i+2 because positions start at 1

        # Calculate ideal DCG
        ideal_retrieved = relevant[:k]
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def average_precision(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Average Precision"""
        relevant_set = set(relevant)
        precisions = []
        hits = 0

        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                hits += 1
                precisions.append(hits / (i + 1))

        if not precisions:
            return 0.0

        return sum(precisions) / len(relevant)

    def evaluate_from_file(
        self,
        eval_file: str,
        k_values: List[int] = None
    ) -> Tuple[List[EvaluationResult], Dict[str, float]]:
        """
        Evaluate from a JSONL evaluation file.

        Args:
            eval_file: Path to evaluation dataset (JSONL format)
            k_values: List of k values to evaluate at

        Returns:
            Tuple of (individual results, aggregated metrics)
        """
        queries = []
        relevant_docs = []

        # Load evaluation dataset
        with open(eval_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    queries.append(data['query'])
                    relevant_docs.append(data.get('relevant_docs', data.get('relevant', [])))

        # Run evaluation
        results = self.evaluate(queries, relevant_docs, k_values)

        # Aggregate results
        aggregated = self.aggregate_results(results)

        return results, aggregated

    @staticmethod
    def aggregate_results(results: List[EvaluationResult]) -> Dict[str, float]:
        """Aggregate evaluation results"""
        if not results:
            return {}

        # Group by k value
        k_groups = {}
        for result in results:
            k = result.k
            if k not in k_groups:
                k_groups[k] = []
            k_groups[k].append(result)

        aggregated = {}
        for k, group in k_groups.items():
            # Average metrics for each k
            metrics_list = [r.metrics for r in group]
            metric_names = metrics_list[0].keys() if metrics_list else []

            for metric in metric_names:
                values = [m[metric] for m in metrics_list if metric in m]
                if values:
                    key = f"{metric}@{k}"
                    aggregated[key] = np.mean(values)

        return aggregated

    def save_results(
        self,
        results: List[EvaluationResult],
        output_path: str
    ):
        """Save evaluation results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        serializable = []
        for result in results:
            serializable.append({
                'query': result.query,
                'retrieved': result.retrieved,
                'relevant': result.relevant,
                'metrics': result.metrics,
                'k': result.k
            })

        # Save as JSONL
        with open(output_path, 'w') as f:
            for item in serializable:
                f.write(json.dumps(item) + '\n')

        logger.info(f"Saved {len(results)} evaluation results to {output_path}")

    def print_summary(self, aggregated: Dict[str, float]):
        """Print evaluation summary"""
        print("\n" + "=" * 50)
        print("Evaluation Summary")
        print("=" * 50)

        # Group by metric
        metrics_by_name = {}
        for key, value in aggregated.items():
            if '@' in key:
                metric, k = key.rsplit('@', 1)
                if metric not in metrics_by_name:
                    metrics_by_name[metric] = {}
                metrics_by_name[metric][int(k)] = value

        # Print each metric
        for metric, k_values in metrics_by_name.items():
            print(f"\n{metric.upper()}:")
            for k in sorted(k_values.keys()):
                print(f"  @{k:2d}: {k_values[k]:.4f}")

        print("=" * 50)