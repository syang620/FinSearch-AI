#!/usr/bin/env python3
"""
Evaluation script for BM25-only retrieval
Tests keyword/lexical search performance without vector embeddings
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import sys
sys.path.insert(0, '../backend')

from app.services.rag.bm25_index import bm25_index


def evaluate_bm25_retrieval(
    question: str,
    expected_sources: List[str],
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Evaluate BM25 retrieval for a single question
    """
    start_time = time.time()

    # Search using BM25 only
    results = bm25_index.search(
        query=question,
        n_results=n_results,
        where={'ticker': 'AAPL'}  # BM25 uses 'ticker' not 'company'
    )

    latency = (time.time() - start_time) * 1000  # Convert to ms

    # Extract retrieved sources
    retrieved_sources = []
    if results['ids'] and results['ids'][0]:
        for metadata in results['metadatas'][0]:
            source = metadata.get('filename') or metadata.get('source', 'Unknown')
            retrieved_sources.append(source)

    # Calculate metrics
    hits = [1 if src in expected_sources else 0 for src in retrieved_sources]

    # Hit Rate at K
    hit_at_1 = hits[0] if len(hits) > 0 else 0
    hit_at_3 = 1 if sum(hits[:3]) > 0 else 0
    hit_at_5 = 1 if sum(hits[:5]) > 0 else 0

    # Mean Reciprocal Rank (MRR)
    mrr = 0
    for i, hit in enumerate(hits):
        if hit:
            mrr = 1.0 / (i + 1)
            break

    # Precision at K
    precision_at_1 = hits[0] if len(hits) > 0 else 0
    precision_at_3 = sum(hits[:3]) / min(3, len(hits)) if hits else 0
    precision_at_5 = sum(hits[:5]) / min(5, len(hits)) if hits else 0

    return {
        'question': question,
        'expected_sources': expected_sources,
        'retrieved_sources': retrieved_sources,
        'hit_at_1': hit_at_1,
        'hit_at_3': hit_at_3,
        'hit_at_5': hit_at_5,
        'mrr': mrr,
        'precision_at_1': precision_at_1,
        'precision_at_3': precision_at_3,
        'precision_at_5': precision_at_5,
        'latency_ms': latency,
        'num_retrieved': len(retrieved_sources)
    }


def main():
    # Load evaluation dataset
    dataset_path = Path("retrieval_eval_dataset.json")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Handle both list and dict formats
    if isinstance(dataset, list):
        questions = dataset
    else:
        questions = dataset.get('questions', dataset)

    print(f"Loaded {len(questions)} evaluation questions")
    print("\nStarting BM25-only evaluation...")
    print("=" * 60)

    all_results = []

    # Track overall metrics
    total_mrr = 0
    total_hit_1 = 0
    total_hit_3 = 0
    total_hit_5 = 0
    total_latency = 0
    category_metrics = {}

    # Evaluate each question
    for i, q in enumerate(questions, 1):
        result = evaluate_bm25_retrieval(
            question=q['question'],
            expected_sources=q['filings']  # Dataset uses 'filings' for expected sources
        )
        result['id'] = q['id']
        result['category'] = q['category']
        all_results.append(result)

        # Update totals
        total_mrr += result['mrr']
        total_hit_1 += result['hit_at_1']
        total_hit_3 += result['hit_at_3']
        total_hit_5 += result['hit_at_5']
        total_latency += result['latency_ms']

        # Track by category
        cat = q['category']
        if cat not in category_metrics:
            category_metrics[cat] = {
                'count': 0, 'mrr': 0, 'hit_5': 0
            }
        category_metrics[cat]['count'] += 1
        category_metrics[cat]['mrr'] += result['mrr']
        category_metrics[cat]['hit_5'] += result['hit_at_5']

        # Progress indicator
        print(f"[{i:2}/{len(questions)}] Evaluating: {q['id'][:35]:35} ... "
              f"MRR={result['mrr']:.2f} Hit@5={result['hit_at_5']}")

    # Calculate averages
    n = len(questions)
    avg_metrics = {
        'total_questions': n,
        'hit_rate_at_1': (total_hit_1 / n) * 100,
        'hit_rate_at_3': (total_hit_3 / n) * 100,
        'hit_rate_at_5': (total_hit_5 / n) * 100,
        'mrr': total_mrr / n,
        'avg_latency_ms': total_latency / n,
        'category_breakdown': {}
    }

    # Calculate category averages
    for cat, metrics in category_metrics.items():
        avg_metrics['category_breakdown'][cat] = {
            'questions': metrics['count'],
            'hit_rate_at_5': (metrics['hit_5'] / metrics['count']) * 100,
            'mrr': metrics['mrr'] / metrics['count']
        }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"bm25_evaluation_results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'summary': avg_metrics,
            'detailed_results': all_results
        }, f, indent=2)

    print("=" * 60)
    print("BM25-Only Evaluation Complete!")
    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 60)
    print("BM25-ONLY EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nTotal Questions: {n}")
    print(f"\nRETRIEVAL METRICS:")
    print(f"  Hit Rate@1: {avg_metrics['hit_rate_at_1']:.1f}%")
    print(f"  Hit Rate@3: {avg_metrics['hit_rate_at_3']:.1f}%")
    print(f"  Hit Rate@5: {avg_metrics['hit_rate_at_5']:.1f}%")
    print(f"  MRR: {avg_metrics['mrr']:.3f}")
    print(f"  Avg Latency: {avg_metrics['avg_latency_ms']:.0f}ms")

    print(f"\nCATEGORY BREAKDOWN:")
    for cat, metrics in avg_metrics['category_breakdown'].items():
        print(f"\n  {cat}:")
        print(f"    Questions: {metrics['questions']}")
        print(f"    Hit Rate@5: {metrics['hit_rate_at_5']:.1f}%")
        print(f"    MRR: {metrics['mrr']:.3f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()