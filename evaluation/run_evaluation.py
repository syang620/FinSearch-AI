#!/usr/bin/env python3
"""
Simple evaluation script for FinSearch AI retrieval metrics
Runs queries from the evaluation dataset and calculates key metrics
"""

import json
import sys
import time
from typing import List, Dict, Any
from datetime import datetime
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.rag.retriever import rag_retriever


class SimpleEvaluator:
    """Simple evaluator for retrieval metrics"""

    def __init__(self, dataset_path: str):
        """Initialize with evaluation dataset"""
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        self.results = []
        print(f"Loaded {len(self.dataset)} evaluation questions")

    def evaluate_question(self, question_data: Dict) -> Dict:
        """Evaluate a single question"""
        start_time = time.time()

        try:
            # Retrieve context using RAG retriever
            retrieved = rag_retriever.retrieve_context(
                query=question_data['question'],
                n_results=5,
                company_filter='AAPL'  # All questions are about Apple
            )

            latency = time.time() - start_time

            # Extract retrieved sources and texts
            retrieved_sources = []
            retrieved_texts = []

            if retrieved and 'contexts' in retrieved:
                for ctx in retrieved['contexts']:
                    # Try multiple metadata fields for source
                    metadata = ctx.get('metadata', {})
                    source = metadata.get('source_uri') or metadata.get('filename') or metadata.get('source', 'Unknown')
                    # Extract just filename if it's a path
                    if '/' in source:
                        source = source.split('/')[-1]
                    retrieved_sources.append(source)
                    retrieved_texts.append(ctx.get('text', ''))

            # Get expected sources from ground truth
            expected_sources = []
            for filing in question_data.get('filings', []):
                source = filing.get('source', '')
                # Convert to expected filename format (e.g., "2025 Form 10-K" -> variations)
                expected_sources.append(source)
                # Add variations of the source name
                if '10-K' in source:
                    year = source.split()[0] if source[0].isdigit() else ''
                    if year:
                        expected_sources.append(f"AAPL_{year}_10-K.pdf")
                        expected_sources.append(f"AAPL_FY{year}_10-K.pdf")
                elif '10-Q' in source:
                    # Extract year and quarter
                    parts = source.split()
                    year = parts[0] if parts and parts[0].isdigit() else ''
                    quarter = source.split('Q')[-1].strip() if 'Q' in source else ''
                    if year and quarter:
                        expected_sources.append(f"AAPL_{year}_Q{quarter}.pdf")
                        expected_sources.append(f"AAPL_FY{year}_Q{quarter}.pdf")

            # Calculate metrics
            metrics = self.calculate_metrics(
                retrieved_sources,
                expected_sources,
                retrieved_texts,
                question_data.get('answer', {})
            )

            metrics['latency_ms'] = latency * 1000
            metrics['question_id'] = question_data['id']
            metrics['category'] = question_data['category']
            metrics['retrieved_sources'] = retrieved_sources[:3]  # Store top 3 for analysis

        except Exception as e:
            print(f"    Error evaluating {question_data['id']}: {str(e)}")
            metrics = {
                'question_id': question_data['id'],
                'category': question_data['category'],
                'error': str(e),
                'hit_rate@1': 0.0,
                'hit_rate@3': 0.0,
                'hit_rate@5': 0.0,
                'mrr': 0.0,
                'precision@1': 0.0,
                'precision@3': 0.0,
                'precision@5': 0.0,
                'answer_coverage': 0.0,
                'latency_ms': 0.0
            }

        return metrics

    def calculate_metrics(self, retrieved: List[str], expected: List[str],
                         texts: List[str], answer: Dict) -> Dict:
        """Calculate retrieval metrics"""
        metrics = {}

        # Normalize for comparison (case-insensitive)
        retrieved_lower = [r.lower() for r in retrieved]
        expected_lower = [e.lower() for e in expected]

        # Hit Rate @ K (did we find at least one relevant document?)
        for k in [1, 3, 5]:
            top_k = retrieved_lower[:k]
            hit = any(
                any(exp in ret or ret in exp for exp in expected_lower)
                for ret in top_k
            )
            metrics[f'hit_rate@{k}'] = 1.0 if hit else 0.0

        # MRR (Mean Reciprocal Rank) - where's the first relevant doc?
        mrr = 0.0
        for i, ret in enumerate(retrieved_lower):
            if any(exp in ret or ret in exp for exp in expected_lower):
                mrr = 1.0 / (i + 1)
                break
        metrics['mrr'] = mrr

        # Precision @ K (what fraction of retrieved docs are relevant?)
        for k in [1, 3, 5]:
            top_k = retrieved_lower[:k]
            if len(top_k) > 0:
                relevant = sum(
                    1 for ret in top_k
                    if any(exp in ret or ret in exp for exp in expected_lower)
                )
                metrics[f'precision@{k}'] = relevant / len(top_k)
            else:
                metrics[f'precision@{k}'] = 0.0

        # Answer Coverage (does the retrieved text contain the answer?)
        answer_found = False
        if answer and 'value' in answer:
            answer_value = answer['value']
            combined_text = ' '.join(texts).lower()

            # Handle different answer types
            if isinstance(answer_value, (int, float)):
                # For numeric answers, check if the number appears
                answer_str = str(answer_value)
                # Also check with commas for large numbers
                answer_with_comma = f"{answer_value:,}" if isinstance(answer_value, int) else str(answer_value)
                answer_found = (
                    answer_str in combined_text or
                    answer_with_comma in combined_text or
                    answer_str.replace('.0', '') in combined_text  # Handle float to int
                )
            elif isinstance(answer_value, str):
                # For text answers, check substring
                answer_found = answer_value.lower() in combined_text
            elif isinstance(answer_value, list):
                # For list answers, check if any item is found
                answer_found = any(
                    str(item).lower() in combined_text
                    for item in answer_value
                )
            elif isinstance(answer_value, dict):
                # For structured answers, check if any value is found
                answer_found = any(
                    str(v).lower() in combined_text
                    for v in answer_value.values()
                    if v is not None
                )

        metrics['answer_coverage'] = 1.0 if answer_found else 0.0

        return metrics

    def run_evaluation(self) -> Dict:
        """Run evaluation on entire dataset"""
        print(f"\nStarting evaluation of {len(self.dataset)} questions...")
        print("=" * 60)

        category_results = {}
        all_metrics = []

        for i, question in enumerate(self.dataset):
            print(f"[{i+1:2d}/{len(self.dataset)}] Evaluating: {question['id'][:30]:30s}", end=' ... ')

            result = self.evaluate_question(question)
            all_metrics.append(result)

            # Group by category
            category = question['category']
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)

            # Print quick result
            if 'error' not in result:
                print(f"MRR={result['mrr']:.2f} Hit@5={result['hit_rate@5']:.0f}")
            else:
                print("ERROR")

        print("=" * 60)
        print("Evaluation complete!\n")

        # Calculate aggregates
        aggregate_metrics = self.calculate_aggregates(all_metrics)
        category_aggregates = {
            cat: self.calculate_aggregates(results)
            for cat, results in category_results.items()
        }

        return {
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(self.dataset),
            'successful_evaluations': len([r for r in all_metrics if 'error' not in r]),
            'aggregate_metrics': aggregate_metrics,
            'category_metrics': category_aggregates,
            'detailed_results': all_metrics
        }

    def calculate_aggregates(self, results: List[Dict]) -> Dict:
        """Calculate average metrics"""
        if not results:
            return {}

        # Filter out results with errors
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            return {'error': 'No valid results'}

        metrics = {}
        metric_names = [
            'hit_rate@1', 'hit_rate@3', 'hit_rate@5',
            'mrr',
            'precision@1', 'precision@3', 'precision@5',
            'answer_coverage', 'latency_ms'
        ]

        for metric in metric_names:
            values = [r.get(metric, 0) for r in valid_results]
            if values:
                metrics[f'avg_{metric}'] = sum(values) / len(values)
                metrics[f'max_{metric}'] = max(values)
                metrics[f'min_{metric}'] = min(values)

        metrics['total_questions'] = len(results)
        metrics['successful_questions'] = len(valid_results)

        return metrics

    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

    def print_summary(self, results: Dict):
        """Print a summary of the results"""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        agg = results['aggregate_metrics']

        print(f"\nTotal Questions: {results['total_questions']}")
        print(f"Successful Evaluations: {results['successful_evaluations']}")

        print(f"\nRETRIEVAL METRICS:")
        print(f"  Hit Rate@1: {agg.get('avg_hit_rate@1', 0):.1%}")
        print(f"  Hit Rate@3: {agg.get('avg_hit_rate@3', 0):.1%}")
        print(f"  Hit Rate@5: {agg.get('avg_hit_rate@5', 0):.1%}")
        print(f"  MRR: {agg.get('avg_mrr', 0):.3f}")
        print(f"  Precision@1: {agg.get('avg_precision@1', 0):.1%}")
        print(f"  Precision@3: {agg.get('avg_precision@3', 0):.1%}")
        print(f"  Precision@5: {agg.get('avg_precision@5', 0):.1%}")

        print(f"\nCONTENT METRICS:")
        print(f"  Answer Coverage: {agg.get('avg_answer_coverage', 0):.1%}")

        print(f"\nPERFORMANCE:")
        print(f"  Avg Latency: {agg.get('avg_latency_ms', 0):.0f}ms")
        print(f"  Min Latency: {agg.get('min_latency_ms', 0):.0f}ms")
        print(f"  Max Latency: {agg.get('max_latency_ms', 0):.0f}ms")

        print(f"\nCATEGORY BREAKDOWN:")
        for category, cat_metrics in results['category_metrics'].items():
            print(f"\n  {category}:")
            print(f"    Questions: {cat_metrics.get('total_questions', 0)}")
            print(f"    Hit Rate@5: {cat_metrics.get('avg_hit_rate@5', 0):.1%}")
            print(f"    MRR: {cat_metrics.get('avg_mrr', 0):.3f}")
            print(f"    Answer Coverage: {cat_metrics.get('avg_answer_coverage', 0):.1%}")

        print("\n" + "=" * 60)


def main():
    """Main execution"""
    # Check if dataset exists
    dataset_path = os.path.join(os.path.dirname(__file__), 'retrieval_eval_dataset.json')
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    # Create evaluator and run
    evaluator = SimpleEvaluator(dataset_path)
    results = evaluator.run_evaluation()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        os.path.dirname(__file__),
        f'evaluation_results_{timestamp}.json'
    )
    evaluator.save_results(results, output_path)

    # Print summary
    evaluator.print_summary(results)

    return results


if __name__ == "__main__":
    main()