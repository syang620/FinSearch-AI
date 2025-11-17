"""
Unified evaluation module for RAG system.
Handles both retrieval-only and end-to-end (retrieval + generation) evaluation.
"""

import os
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import requests
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Unified evaluator for RAG pipeline - handles both retrieval and generation evaluation"""

    def __init__(self, top_k: int = 5, with_generation: bool = False):
        """
        Initialize RAG evaluator.

        Args:
            top_k: Number of documents to retrieve
            with_generation: Whether to evaluate generation (requires LLM)
        """
        logger.info("Initializing RAG Evaluator...")

        # Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(path='data/chroma_db')
        self.collection = self.chroma_client.get_collection('financial_documents')

        # Initialize embedder and reranker
        self.embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

        self.top_k = top_k
        self.with_generation = with_generation
        self.llm_model = None

        print(f"✓ Initialized RAG evaluator (k={top_k})")
        print(f"✓ Connected to ChromaDB ({self.collection.count():,} documents)")

        if with_generation:
            self._init_llm()
            if self.llm_model:
                print(f"✓ Generation enabled with {self.llm_model}")
            else:
                print("⚠ Generation requested but no LLM available - retrieval only")
                self.with_generation = False
        else:
            print("✓ Retrieval-only evaluation mode")

    def _init_llm(self):
        """Initialize LLM for generation evaluation"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])

                # Prefer gemma3:270m for fast evaluation
                for model in models:
                    if 'gemma3:270m' in model['name']:
                        self.llm_model = model['name']
                        break

                if not self.llm_model and models:
                    self.llm_model = models[0]['name']

        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")

    def retrieve_and_rerank(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Retrieve and rerank documents for a query"""
        if k is None:
            k = self.top_k

        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()

        # Retrieve from ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k * 3, 30)  # Get 3x for reranking
        )

        if not results['documents'][0]:
            return []

        # Prepare for reranking
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results else [0] * len(documents)

        # Rerank
        pairs = [[query, doc] for doc in documents]
        rerank_scores = self.reranker.predict(pairs)

        # Combine results
        ranked_results = []
        for idx, score in enumerate(rerank_scores):
            ranked_results.append({
                'text': documents[idx],
                'rerank_score': float(score),
                'vector_distance': float(distances[idx]),
                'metadata': metadatas[idx],
                'rank': 0
            })

        # Sort by rerank score
        ranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Add ranks
        for i, result in enumerate(ranked_results[:k]):
            result['rank'] = i + 1

        return ranked_results[:k]

    def generate_answer(self, query: str, context: List[Dict]) -> Optional[str]:
        """Generate answer using LLM (if enabled)"""
        if not self.with_generation or not self.llm_model or not context:
            return None

        # Prepare context
        context_text = "\n\n".join([
            f"[Document {i+1}]\n{doc['text'][:500]}"
            for i, doc in enumerate(context[:3])
        ])

        prompt = f"""You are a financial analyst assistant. Answer the question based on the provided context.

Context:
{context_text}

Question: {query}

Answer: """

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.llm_model,
                    'prompt': prompt,
                    'stream': False
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json().get('response', '')
        except Exception as e:
            logger.error(f"Generation error: {e}")

        return None

    def evaluate_single(self, item: Dict) -> Dict:
        """Evaluate a single question"""
        question_id = item.get('id', 'unknown')
        question = item['question']
        expected_answer = item.get('answer', {})
        expected_filings = item.get('filings', [])  # Get expected source documents
        category = item.get('category', 'unknown')

        print(f"\rEvaluating: {question_id} ({category})", end='')

        # Retrieval evaluation
        retrieval_start = time.time()
        retrieved_docs = self.retrieve_and_rerank(question)
        retrieval_time = time.time() - retrieval_start

        # Calculate retrieval metrics - now includes expected sources
        retrieval_metrics = self._calculate_retrieval_metrics(
            retrieved_docs, expected_answer, question, expected_filings
        )

        # Generation evaluation (if enabled)
        generation_metrics = {}
        generated_answer = None

        if self.with_generation and self.llm_model:
            generation_start = time.time()
            generated_answer = self.generate_answer(question, retrieved_docs)
            generation_time = time.time() - generation_start

            if generated_answer:
                generation_metrics = self._calculate_generation_metrics(
                    generated_answer, expected_answer, question
                )
                generation_metrics['generation_time_ms'] = round(generation_time * 1000, 2)
                generation_metrics['generated'] = True
            else:
                generation_metrics['generated'] = False

        # Format retrieved documents with source matching info
        retrieved_formatted = []
        expected_sources = set()
        for filing in expected_filings:
            source = filing.get('source', '')
            expected_sources.add(source)
            expected_sources.add(source.replace('raw_', '').replace('.htm', '.txt'))
            expected_sources.add(source.replace('raw_', '').replace('.htm', ''))

        for doc in retrieved_docs[:5]:  # Show top 5 for better analysis
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')

            # Check if this matches expected source
            is_correct = False
            normalized_retrieved = filename.replace('raw_', '').replace('.htm', '').replace('.txt', '')
            for expected in expected_sources:
                normalized_expected = expected.replace('raw_', '').replace('.htm', '').replace('.txt', '')
                if normalized_expected and (normalized_expected == normalized_retrieved or
                                          normalized_expected in normalized_retrieved or
                                          normalized_retrieved in normalized_expected):
                    is_correct = True
                    break

            retrieved_formatted.append({
                'rank': doc['rank'],
                'company': metadata.get('company', 'Unknown'),
                'doc_type': metadata.get('document_type', 'Unknown'),
                'filename': filename,
                'year': metadata.get('year', 'Unknown'),
                'rerank_score': round(doc['rerank_score'], 4),
                'is_correct_source': is_correct,
                'text_snippet': doc['text'][:200] + '...' if len(doc['text']) > 200 else doc['text']
            })

        result = {
            'id': question_id,
            'question': question,
            'category': category,
            'expected_answer': expected_answer,
            'expected_sources': expected_filings,  # Add expected source info
            'retrieved_documents': retrieved_formatted,
            'retrieval_metrics': {
                'retrieval_time_ms': round(retrieval_time * 1000, 2),
                'num_retrieved': len(retrieved_docs),
                **retrieval_metrics
            }
        }

        # Add generation results if available
        if self.with_generation:
            result['generated_answer'] = generated_answer[:500] if generated_answer else None
            result['generation_metrics'] = generation_metrics

        return result

    def _calculate_retrieval_metrics(self, retrieved_docs: List[Dict], expected_answer: Dict,
                                    question: str, expected_filings: List[Dict]) -> Dict:
        """Calculate retrieval-specific metrics based on expected source documents"""
        if not retrieved_docs:
            return {
                'has_correct_source': False,
                'top_1_correct': False,
                'top_3_correct': False,
                'top_5_correct': False,
                'hit_rate': 0.0,
                'mrr': 0.0,
                'avg_rerank_score': 0.0,
                'company_match': False,
                'source_accuracy': 0.0
            }

        # Extract expected source filenames
        expected_sources = set()
        for filing in expected_filings:
            source = filing.get('source', '')
            expected_sources.add(source)
            # Also add normalized versions
            expected_sources.add(source.replace('raw_', '').replace('.htm', '.txt'))
            expected_sources.add(source.replace('raw_', '').replace('.htm', ''))

        # Check each retrieved document
        correct_source_ranks = []
        company_matches = []

        for i, doc in enumerate(retrieved_docs[:5], 1):  # Check top 5
            metadata = doc.get('metadata', {})
            retrieved_filename = metadata.get('filename', '')
            retrieved_company = metadata.get('company', '')

            # Normalize filename for comparison
            normalized_retrieved = retrieved_filename.replace('raw_', '').replace('.htm', '').replace('.txt', '')

            # Check if this is a correct source document
            is_correct_source = False
            for expected in expected_sources:
                normalized_expected = expected.replace('raw_', '').replace('.htm', '').replace('.txt', '')
                if normalized_expected and (normalized_expected == normalized_retrieved or
                                           normalized_expected in normalized_retrieved or
                                           normalized_retrieved in normalized_expected):
                    is_correct_source = True
                    correct_source_ranks.append(i)
                    break

            # Check company match (should be Apple/AAPL)
            if 'apple' in retrieved_company.lower() or retrieved_company == 'AAPL':
                company_matches.append(i)

        # Calculate MRR (Mean Reciprocal Rank)
        mrr = 0.0
        if correct_source_ranks:
            mrr = 1.0 / correct_source_ranks[0]  # Use first correct source rank

        # Calculate hit rate (percentage of correct sources in top 5)
        hit_rate = len(correct_source_ranks) / min(5, len(retrieved_docs)) if retrieved_docs else 0

        # Source accuracy - how many of top 3 are correct sources
        source_accuracy = len([r for r in correct_source_ranks if r <= 3]) / min(3, len(retrieved_docs))

        return {
            'has_correct_source': len(correct_source_ranks) > 0,
            'top_1_correct': 1 in correct_source_ranks,
            'top_3_correct': any(r <= 3 for r in correct_source_ranks),
            'top_5_correct': any(r <= 5 for r in correct_source_ranks),
            'hit_rate': round(hit_rate * 100, 1),
            'mrr': round(mrr, 3),
            'avg_rerank_score': round(float(np.mean([d['rerank_score'] for d in retrieved_docs[:5]])), 4),
            'company_match': len(company_matches) > 0,
            'source_accuracy': round(source_accuracy * 100, 1),
            'correct_source_ranks': correct_source_ranks[:3] if correct_source_ranks else []
        }

    def _calculate_generation_metrics(self, generated: str, expected: Dict, question: str) -> Dict:
        """Calculate generation-specific metrics"""
        if not generated:
            return {'answer_present': False, 'correct_value': False}

        generated_lower = generated.lower()

        # Check if expected value appears in generation
        expected_value = str(expected.get('value', ''))
        answer_present = expected_value.lower() in generated_lower

        # Check for numeric match (with some tolerance)
        correct_value = False
        if expected.get('type') == 'numeric' and expected_value:
            # Look for the number in the generated text
            import re
            numbers = re.findall(r'[\d,]+\.?\d*', generated)
            for num_str in numbers:
                try:
                    num = float(num_str.replace(',', ''))
                    expected_num = float(expected_value)
                    # Allow 5% tolerance for numeric answers
                    if abs(num - expected_num) / expected_num < 0.05:
                        correct_value = True
                        break
                except:
                    pass

        return {
            'answer_present': answer_present,
            'correct_value': correct_value,
            'response_length': len(generated)
        }

    def evaluate_dataset(self, dataset_path: str) -> Dict:
        """Evaluate entire dataset"""
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        mode = "End-to-End (Retrieval + Generation)" if self.with_generation else "Retrieval-Only"
        print(f"\nEvaluating {len(dataset)} questions in {mode} mode")
        print(f"Top-k: {self.top_k}")
        print("-" * 60)

        results = []
        for item in dataset:
            result = self.evaluate_single(item)
            results.append(result)

        print("\n" + "=" * 60)

        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)

        return {
            'evaluation_config': {
                'dataset': dataset_path,
                'mode': mode,
                'top_k': self.top_k,
                'embedder': 'BAAI/bge-base-en-v1.5',
                'reranker': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
                'llm': self.llm_model if self.with_generation else None,
                'num_questions': len(dataset)
            },
            'aggregate_metrics': aggregate_metrics,
            'results': results
        }

    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all queries"""
        total = len(results)
        if total == 0:
            return {}

        # Retrieval metrics
        retrieval_metrics_list = [r['retrieval_metrics'] for r in results]

        aggregate = {
            'total_questions': total,
            'retrieval': {
                'avg_time_ms': round(np.mean([m['retrieval_time_ms'] for m in retrieval_metrics_list]), 2),
                'avg_docs_retrieved': round(np.mean([m['num_retrieved'] for m in retrieval_metrics_list]), 1),
                'pct_with_correct_source': round(sum(m['has_correct_source'] for m in retrieval_metrics_list) / total * 100, 1),
                'pct_top_1_correct': round(sum(m['top_1_correct'] for m in retrieval_metrics_list) / total * 100, 1),
                'pct_top_3_correct': round(sum(m['top_3_correct'] for m in retrieval_metrics_list) / total * 100, 1),
                'pct_top_5_correct': round(sum(m['top_5_correct'] for m in retrieval_metrics_list) / total * 100, 1),
                'avg_hit_rate': round(np.mean([m['hit_rate'] for m in retrieval_metrics_list]), 1),
                'avg_mrr': round(np.mean([m['mrr'] for m in retrieval_metrics_list]), 3),
                'avg_rerank_score': round(np.mean([m['avg_rerank_score'] for m in retrieval_metrics_list]), 4),
                'pct_company_match': round(sum(m['company_match'] for m in retrieval_metrics_list) / total * 100, 1),
                'avg_source_accuracy': round(np.mean([m['source_accuracy'] for m in retrieval_metrics_list]), 1)
            }
        }

        # Generation metrics (if available)
        if self.with_generation:
            generation_results = [r for r in results if 'generation_metrics' in r and r['generation_metrics'].get('generated')]
            if generation_results:
                gen_metrics_list = [r['generation_metrics'] for r in generation_results]
                aggregate['generation'] = {
                    'pct_generated': round(len(generation_results) / total * 100, 1),
                    'avg_time_ms': round(np.mean([m['generation_time_ms'] for m in gen_metrics_list]), 2),
                    'pct_answer_present': round(sum(m['answer_present'] for m in gen_metrics_list) / len(gen_metrics_list) * 100, 1),
                    'pct_correct_value': round(sum(m['correct_value'] for m in gen_metrics_list) / len(gen_metrics_list) * 100, 1),
                    'avg_response_length': round(np.mean([m['response_length'] for m in gen_metrics_list]), 0)
                }
                aggregate['total_avg_time_ms'] = round(
                    aggregate['retrieval']['avg_time_ms'] + aggregate['generation']['avg_time_ms'], 2
                )

        # Category breakdown
        categories = {}
        for result in results:
            cat = result.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)

        if len(categories) > 1:
            aggregate['by_category'] = {}
            for cat, cat_results in categories.items():
                cat_retrieval = [r['retrieval_metrics'] for r in cat_results]
                cat_metrics = {
                    'count': len(cat_results),
                    'retrieval': {
                        'pct_correct_source': round(sum(m['has_correct_source'] for m in cat_retrieval) / len(cat_retrieval) * 100, 1),
                        'avg_mrr': round(np.mean([m['mrr'] for m in cat_retrieval]), 3),
                        'avg_hit_rate': round(np.mean([m['hit_rate'] for m in cat_retrieval]), 1),
                        'avg_score': round(np.mean([m['avg_rerank_score'] for m in cat_retrieval]), 4)
                    }
                }

                if self.with_generation:
                    cat_gen = [r['generation_metrics'] for r in cat_results if 'generation_metrics' in r and r['generation_metrics'].get('generated')]
                    if cat_gen:
                        cat_metrics['generation'] = {
                            'pct_correct': round(sum(m['correct_value'] for m in cat_gen) / len(cat_gen) * 100, 1)
                        }

                aggregate['by_category'][cat] = cat_metrics

        return aggregate

    def print_summary(self, evaluation_result: Dict):
        """Print evaluation summary"""
        config = evaluation_result['evaluation_config']
        metrics = evaluation_result['aggregate_metrics']

        print("\n" + "=" * 60)
        print("RAG EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\nConfiguration:")
        print(f"  Mode: {config['mode']}")
        print(f"  Dataset: {config['dataset']}")
        print(f"  Questions: {config['num_questions']}")
        print(f"  Top-K: {config['top_k']}")
        if config['llm']:
            print(f"  LLM: {config['llm']}")

        print(f"\nRetrieval Performance:")
        ret = metrics['retrieval']
        print(f"  Avg Time: {ret['avg_time_ms']}ms")
        print(f"  Has Correct Source: {ret['pct_with_correct_source']}%")
        print(f"  Top-1 Correct: {ret['pct_top_1_correct']}%")
        print(f"  Top-3 Correct: {ret['pct_top_3_correct']}%")
        print(f"  Top-5 Correct: {ret['pct_top_5_correct']}%")
        print(f"  Avg Hit Rate: {ret['avg_hit_rate']}%")
        print(f"  Avg MRR: {ret['avg_mrr']}")
        print(f"  Company Match: {ret['pct_company_match']}%")
        print(f"  Source Accuracy: {ret['avg_source_accuracy']}%")
        print(f"  Avg Rerank Score: {ret['avg_rerank_score']}")

        if 'generation' in metrics:
            print(f"\nGeneration Performance:")
            gen = metrics['generation']
            print(f"  Successfully Generated: {gen['pct_generated']}%")
            print(f"  Avg Time: {gen['avg_time_ms']}ms")
            print(f"  Answer Present: {gen['pct_answer_present']}%")
            print(f"  Correct Value: {gen['pct_correct_value']}%")
            print(f"  Avg Response Length: {gen['avg_response_length']:.0f} chars")
            print(f"\nTotal Pipeline Time: {metrics['total_avg_time_ms']}ms")

        if 'by_category' in metrics:
            print(f"\nPerformance by Category:")
            for cat, cat_metrics in metrics['by_category'].items():
                print(f"  {cat} ({cat_metrics['count']} questions):")
                print(f"    Correct Source: {cat_metrics['retrieval']['pct_correct_source']}%")
                print(f"    MRR: {cat_metrics['retrieval']['avg_mrr']}")
                print(f"    Hit Rate: {cat_metrics['retrieval']['avg_hit_rate']}%")
                if 'generation' in cat_metrics:
                    print(f"    Generation: {cat_metrics['generation']['pct_correct']}% correct")

        print("=" * 60)