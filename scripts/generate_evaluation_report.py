#!/usr/bin/env python3
"""
Generate detailed evaluation report from evaluation results.
Creates a markdown report similar to evaluation_review.md format.
"""

import json
import os
from datetime import datetime
from pathlib import Path


def load_evaluation_results(results_path):
    """Load the evaluation results JSON"""
    with open(results_path, 'r') as f:
        return json.load(f)


def calculate_metrics(results):
    """Calculate overall and category metrics"""
    metrics = {
        'total': len(results['results']),
        'successful': 0,
        'hit_rate': 0,
        'mrr': 0,
        'precision': 0,
        'source_accuracy': 0,
        'avg_latency': 0,
        'by_category': {}
    }

    # Calculate per-question metrics
    for result in results['results']:
        category = result.get('category', 'unknown')
        if category not in metrics['by_category']:
            metrics['by_category'][category] = {
                'questions': 0,
                'correct_sources': 0,
                'mrr_sum': 0,
                'hit_rate_sum': 0
            }

        metrics['by_category'][category]['questions'] += 1

        # Check if retrieval was successful (correct source found)
        retrieval_metrics = result.get('retrieval_metrics', {})
        if retrieval_metrics.get('has_correct_source'):
            metrics['successful'] += 1
            metrics['by_category'][category]['correct_sources'] += 1

        # Add MRR
        mrr = retrieval_metrics.get('mrr', 0)
        metrics['mrr'] += mrr
        metrics['by_category'][category]['mrr_sum'] += mrr

        # Add hit rate
        hit_rate = retrieval_metrics.get('hit_rate', 0)
        metrics['by_category'][category]['hit_rate_sum'] += hit_rate

        # Source accuracy
        metrics['source_accuracy'] += retrieval_metrics.get('source_accuracy', 0)

        # Latency
        metrics['avg_latency'] += retrieval_metrics.get('retrieval_time_ms', 0)

    # Calculate final metrics
    if metrics['total'] > 0:
        metrics['hit_rate'] = (metrics['successful'] / metrics['total']) * 100
        metrics['mrr'] = metrics['mrr'] / metrics['total']
        metrics['source_accuracy'] = metrics['source_accuracy'] / metrics['total']
        metrics['avg_latency'] = metrics['avg_latency'] / metrics['total']
        metrics['precision'] = metrics['hit_rate'] / 5  # Assuming top-5 retrieval

        # Calculate category metrics
        for cat, cat_data in metrics['by_category'].items():
            if cat_data['questions'] > 0:
                cat_data['hit_rate'] = (cat_data['correct_sources'] / cat_data['questions']) * 100
                cat_data['mrr'] = cat_data['mrr_sum'] / cat_data['questions']
                cat_data['avg_hit_rate'] = cat_data['hit_rate_sum'] / cat_data['questions']

    return metrics


def generate_question_report(result, idx):
    """Generate report for a single question"""
    question_id = result.get('id', f'Question_{idx}')
    question = result['question']
    expected = result.get('expected_answer', {})
    expected_sources = result.get('expected_sources', [])
    category = result.get('category', 'unknown')
    retrieval_metrics = result.get('retrieval_metrics', {})
    retrieved_docs = result.get('retrieved_documents', [])

    # Determine success based on correct source retrieval
    success = "✅" if retrieval_metrics.get('has_correct_source') else "❌"

    # Format expected answer
    if isinstance(expected, dict):
        if 'value' in expected:
            expected_str = f"{expected['value']} {expected.get('unit', '')}"
        else:
            expected_str = str(expected)
    else:
        expected_str = str(expected)

    # Format expected sources
    expected_sources_str = ", ".join([s.get('source', 'Unknown') for s in expected_sources])

    # Build report
    report = f"\n### {idx}. {success} {question_id}\n\n"
    report += f"**Question**: {question}\n\n"
    report += f"**Expected Answer**: {expected_str}\n\n"
    report += f"**Expected Sources**: {expected_sources_str}\n\n"

    # Add retrieved sources
    report += "**Retrieved Documents**:\n"
    for i, doc in enumerate(retrieved_docs[:5], 1):
        company = doc.get('company', 'Unknown')
        filename = doc.get('filename', 'Unknown')
        doc_type = doc.get('doc_type', 'Unknown')
        score = doc.get('rerank_score', 0)
        is_correct = doc.get('is_correct_source', False)

        # Mark if this is a correct source
        correct = " ✓ CORRECT SOURCE" if is_correct else ""
        report += f"  {i}. {company} - {filename} ({doc_type}) - score: {score:.2f}{correct}\n"

    # Add metrics
    report += f"\n**Metrics**:\n"
    report += f"  - Correct Source Found: {'Yes' if retrieval_metrics.get('has_correct_source') else 'No'}\n"
    report += f"  - Top-1 Correct: {'Yes' if retrieval_metrics.get('top_1_correct') else 'No'}\n"
    report += f"  - Top-3 Correct: {'Yes' if retrieval_metrics.get('top_3_correct') else 'No'}\n"
    report += f"  - MRR: {retrieval_metrics.get('mrr', 0):.3f}\n"
    report += f"  - Hit Rate: {retrieval_metrics.get('hit_rate', 0):.1f}%\n"
    report += f"  - Source Accuracy: {retrieval_metrics.get('source_accuracy', 0):.1f}%\n"
    report += f"  - Latency: {retrieval_metrics.get('retrieval_time_ms', 0):.0f}ms\n"

    report += "\n---"

    return report


def generate_markdown_report(results_path, output_path):
    """Generate complete markdown report"""
    results = load_evaluation_results(results_path)
    metrics = calculate_metrics(results)

    # Start report
    report = "# FinSearch AI - Evaluation Results Review\n\n"
    report += f"**Date**: {datetime.now().isoformat()}\n\n"
    report += f"**Total Questions**: {metrics['total']}\n\n"
    report += f"**Successful Evaluations**: {metrics['successful']}\n\n"
    report += "---\n\n"

    # Overall performance
    report += "## Overall Performance\n\n"
    report += f"- **Hit Rate (Correct Source Found)**: {metrics['hit_rate']:.1f}%\n"
    report += f"- **MRR (Mean Reciprocal Rank)**: {metrics['mrr']:.3f}\n"
    report += f"- **Precision@5**: {metrics['precision']:.1f}%\n"
    report += f"- **Source Accuracy**: {metrics['source_accuracy']:.1f}%\n"
    report += f"- **Avg Latency**: {metrics['avg_latency']:.0f}ms\n\n"

    # Performance by category
    report += "## Performance by Category\n\n"
    for cat, cat_metrics in metrics['by_category'].items():
        cat_name = cat.replace('_', ' ').title()
        report += f"### {cat_name}\n"
        report += f"- Questions: {cat_metrics['questions']}\n"
        report += f"- Correct Source Found: {cat_metrics['hit_rate']:.1f}%\n"
        report += f"- MRR: {cat_metrics['mrr']:.3f}\n"
        report += f"- Avg Hit Rate: {cat_metrics['avg_hit_rate']:.1f}%\n\n"

    report += "\n---\n\n"

    # Organize questions by category
    questions_by_category = {}
    for i, result in enumerate(results['results'], 1):
        category = result.get('category', 'unknown')
        if category not in questions_by_category:
            questions_by_category[category] = []
        questions_by_category[category].append((i, result))

    # Generate detailed results by category
    category_names = {
        'single_doc_fact': 'Single Document Fact Questions',
        'single_doc_context': 'Single Document Context Questions',
        'single_doc_multi_period': 'Single Document Multi-Period Questions',
        'cross_doc_fact': 'Cross Document Fact Questions'
    }

    for cat_key, cat_name in category_names.items():
        if cat_key in questions_by_category:
            report += f"\n## {cat_name}\n"
            for idx, result in questions_by_category[cat_key]:
                report += generate_question_report(result, idx)

    # Save report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"✓ Detailed evaluation report saved to {output_path}")


def main():
    """Generate evaluation report"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate evaluation report')
    parser.add_argument('--results', default='data/evaluation/results_retrieval.json',
                       help='Path to evaluation results JSON')
    parser.add_argument('--output', default='data/evaluation/evaluation_results_review.md',
                       help='Output path for markdown report')

    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        return

    generate_markdown_report(args.results, args.output)


if __name__ == "__main__":
    main()