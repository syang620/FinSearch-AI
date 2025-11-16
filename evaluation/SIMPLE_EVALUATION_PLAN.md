# FinSearch AI Simple Evaluation Plan

## Objective
Create a straightforward evaluation pipeline to measure retrieval quality metrics using the 28-question evaluation dataset. Focus on metrics calculation only - analysis will be handled separately.

## Core Metrics to Calculate

### 1. Retrieval Metrics
- **Hit Rate@K** (K=1, 3, 5): Did we retrieve the correct document(s)?
- **MRR (Mean Reciprocal Rank)**: Where does the first relevant document appear?
- **Precision@K**: What percentage of retrieved documents are relevant?
- **Recall@K**: What percentage of relevant documents were retrieved?

### 2. Content Quality Metrics
- **Answer Coverage**: Does the retrieved text contain the answer?
- **Source Accuracy**: Did we identify the correct filing (10-K, 10-Q)?

## Implementation (Single Script)

### File: `evaluation/run_evaluation.py`

```python
"""
Simple evaluation script for FinSearch AI retrieval metrics
"""

import json
import sys
import time
from typing import List, Dict, Any
from datetime import datetime

# Add backend to path
sys.path.append('../backend')
from app.services.rag.retriever import rag_retriever

class SimpleEvaluator:
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        self.results = []

    def evaluate_question(self, question_data: Dict) -> Dict:
        """Evaluate a single question"""
        start_time = time.time()

        # Retrieve context
        retrieved = rag_retriever.retrieve_context(
            query=question_data['question'],
            n_results=5,
            company_filter='AAPL'
        )

        latency = time.time() - start_time

        # Extract retrieved sources
        retrieved_sources = []
        retrieved_texts = []
        if retrieved and 'contexts' in retrieved:
            for ctx in retrieved['contexts']:
                source = ctx.get('metadata', {}).get('filename', 'Unknown')
                retrieved_sources.append(source)
                retrieved_texts.append(ctx.get('text', ''))

        # Get expected sources from ground truth
        expected_sources = [f['source'] for f in question_data['filings']]

        # Calculate metrics
        metrics = self.calculate_metrics(
            retrieved_sources,
            expected_sources,
            retrieved_texts,
            question_data['answer']
        )

        metrics['latency_ms'] = latency * 1000
        metrics['question_id'] = question_data['id']
        metrics['category'] = question_data['category']

        return metrics

    def calculate_metrics(self, retrieved: List[str], expected: List[str],
                         texts: List[str], answer: Dict) -> Dict:
        """Calculate retrieval metrics"""
        metrics = {}

        # Hit Rate @ K
        for k in [1, 3, 5]:
            top_k = retrieved[:k]
            hit = any(exp in str(ret) for exp in expected for ret in top_k)
            metrics[f'hit_rate@{k}'] = 1.0 if hit else 0.0

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, ret in enumerate(retrieved):
            if any(exp in str(ret) for exp in expected):
                mrr = 1.0 / (i + 1)
                break
        metrics['mrr'] = mrr

        # Precision @ K
        for k in [1, 3, 5]:
            top_k = retrieved[:k]
            relevant = sum(1 for ret in top_k if any(exp in str(ret) for exp in expected))
            metrics[f'precision@{k}'] = relevant / k if k > 0 else 0.0

        # Answer Coverage (simple check)
        answer_found = False
        if 'value' in answer:
            answer_str = str(answer['value'])
            combined_text = ' '.join(texts).lower()
            answer_found = answer_str.lower() in combined_text
        metrics['answer_coverage'] = 1.0 if answer_found else 0.0

        return metrics

    def run_evaluation(self) -> Dict:
        """Run evaluation on entire dataset"""
        print(f"Evaluating {len(self.dataset)} questions...")

        category_results = {}
        all_metrics = []

        for i, question in enumerate(self.dataset):
            print(f"  [{i+1}/{len(self.dataset)}] {question['id']}")

            result = self.evaluate_question(question)
            all_metrics.append(result)

            # Group by category
            category = question['category']
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)

        # Calculate aggregates
        aggregate_metrics = self.calculate_aggregates(all_metrics)
        category_aggregates = {
            cat: self.calculate_aggregates(results)
            for cat, results in category_results.items()
        }

        return {
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(self.dataset),
            'aggregate_metrics': aggregate_metrics,
            'category_metrics': category_aggregates,
            'detailed_results': all_metrics
        }

    def calculate_aggregates(self, results: List[Dict]) -> Dict:
        """Calculate average metrics"""
        if not results:
            return {}

        metrics = {}
        metric_names = ['hit_rate@1', 'hit_rate@3', 'hit_rate@5',
                       'mrr', 'precision@1', 'precision@3', 'precision@5',
                       'answer_coverage', 'latency_ms']

        for metric in metric_names:
            values = [r.get(metric, 0) for r in results]
            metrics[f'avg_{metric}'] = sum(values) / len(values)

        return metrics

    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

# Main execution
if __name__ == "__main__":
    evaluator = SimpleEvaluator('retrieval_eval_dataset.json')
    results = evaluator.run_evaluation()
    evaluator.save_results(results, f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    # Print summary
    print("\n=== Evaluation Summary ===")
    agg = results['aggregate_metrics']
    print(f"Hit Rate@5: {agg['avg_hit_rate@5']:.2%}")
    print(f"MRR: {agg['avg_mrr']:.3f}")
    print(f"Precision@5: {agg['avg_precision@5']:.2%}")
    print(f"Answer Coverage: {agg['avg_answer_coverage']:.2%}")
    print(f"Avg Latency: {agg['avg_latency_ms']:.0f}ms")
```

## Output Format

```json
{
  "timestamp": "2024-11-15T10:30:00",
  "total_questions": 28,
  "aggregate_metrics": {
    "avg_hit_rate@1": 0.75,
    "avg_hit_rate@3": 0.85,
    "avg_hit_rate@5": 0.92,
    "avg_mrr": 0.83,
    "avg_precision@1": 0.75,
    "avg_precision@3": 0.65,
    "avg_precision@5": 0.58,
    "avg_answer_coverage": 0.71,
    "avg_latency_ms": 1250
  },
  "category_metrics": {
    "single_doc_fact": { ... },
    "single_doc_context": { ... },
    "single_doc_multi_period": { ... },
    "cross_doc_fact": { ... }
  },
  "detailed_results": [...]
}
```

## Quick Start

1. **Setup**
   ```bash
   cd evaluation/
   pip install -r ../backend/requirements.txt
   ```

2. **Run Evaluation**
   ```bash
   python run_evaluation.py
   ```

3. **View Results**
   - JSON file with all metrics will be created
   - Summary printed to console
   - You handle the analysis

## What This Gives You

- **Raw Metrics**: All the numbers you need for analysis
- **Category Breakdown**: Performance by question type
- **Per-Question Results**: Detailed results for deep dives
- **Simple to Run**: One script, one command
- **Easy to Extend**: Add new metrics as needed

## What This Doesn't Do

- No visualizations (you handle)
- No complex analysis (you handle)
- No recommendations (you handle)
- Just the metrics, clean and simple