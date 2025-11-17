#!/usr/bin/env python3
"""
Unified evaluation script for RAG system.
Supports both retrieval-only and end-to-end evaluation modes.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Set environment for PyTorch
os.environ['USE_TORCH'] = '1'

# Add scripts directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.evaluation import RAGEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG system')

    # Dataset configuration
    parser.add_argument('--dataset', default='data/evaluation/retrieval_eval_dataset.json',
                       help='Path to evaluation dataset')

    # Evaluation mode
    parser.add_argument('--with-generation', action='store_true',
                       help='Include generation evaluation (requires LLM)')
    parser.add_argument('--retrieval-only', action='store_true',
                       help='Retrieval-only evaluation (default)')

    # Retrieval parameters
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of documents to retrieve (default: 5)')

    # Output configuration
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output, show summary only')

    args = parser.parse_args()

    # Determine evaluation mode
    with_generation = args.with_generation and not args.retrieval_only

    # Initialize evaluator
    evaluator = RAGEvaluator(top_k=args.top_k, with_generation=with_generation)

    # Run evaluation
    results = evaluator.evaluate_dataset(args.dataset)

    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'dataset': args.dataset,
        'top_k': args.top_k,
        'mode': 'retrieval+generation' if with_generation else 'retrieval-only'
    }

    # Print summary
    if not args.quiet:
        evaluator.print_summary(results)

    # Save results
    if not args.no_save:
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            mode_suffix = 'full' if with_generation else 'retrieval'
            output_path = f'data/evaluation/results_{mode_suffix}.json'

        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save full results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")

        # Save metrics-only summary
        metrics_output = output_path.replace('.json', '_metrics.json')
        metrics_only = {
            'metadata': results['metadata'],
            'evaluation_config': results['evaluation_config'],
            'aggregate_metrics': results['aggregate_metrics']
        }
        with open(metrics_output, 'w') as f:
            json.dump(metrics_only, f, indent=2)
        print(f"✓ Metrics summary saved to {metrics_output}")

        # Show quick summary if quiet mode
        if args.quiet:
            metrics = results['aggregate_metrics']
            print(f"\nQuick Summary:")
            print(f"  Questions: {metrics['total_questions']}")
            print(f"  Retrieval: {metrics['retrieval']['pct_with_relevant']}% relevant")
            if 'generation' in metrics:
                print(f"  Generation: {metrics['generation']['pct_correct_value']}% correct")


if __name__ == "__main__":
    main()