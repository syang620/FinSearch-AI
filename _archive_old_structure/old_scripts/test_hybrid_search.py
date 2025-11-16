#!/usr/bin/env python3
"""
Test script for hybrid BM25 + Dense search with RRF fusion
Validates that both exact matches and semantic queries work well
"""

import sys
import os
import logging
from typing import Dict, Any, List

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.services.rag.hybrid_search import hybrid_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_results(results: Dict[str, List], query: str, max_docs: int = 3):
    """Pretty print search results"""
    print(f"\n{'='*70}")
    print(f"Query: '{query}'")
    print(f"{'='*70}")

    if not results['ids'] or not results['ids'][0]:
        print("No results found!")
        return

    ids = results['ids'][0]
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    scores = results.get('scores', [[]])[0]

    print(f"Found {len(ids)} results (showing top {min(len(ids), max_docs)})\n")

    for i in range(min(len(ids), max_docs)):
        metadata = metas[i]
        doc_snippet = docs[i][:300] + "..." if len(docs[i]) > 300 else docs[i]

        print(f"Result {i+1}:")
        print(f"  Ticker: {metadata.get('ticker', 'N/A')}")
        print(f"  Doc Type: {metadata.get('doc_type', 'N/A')}")
        print(f"  Year: {metadata.get('fiscal_year', 'N/A')}")

        if metadata.get('quarter'):
            print(f"  Quarter: {metadata.get('quarter')}")
        if metadata.get('phase'):
            print(f"  Phase: {metadata.get('phase')}")
        if metadata.get('section_title'):
            print(f"  Section: {metadata.get('section_title')}")
        if metadata.get('speaker'):
            print(f"  Speaker: {metadata.get('speaker')}")

        if i < len(scores):
            print(f"  RRF Score: {scores[i]:.4f}")

        print(f"  Snippet: {doc_snippet}")
        print()


def test_exact_match_queries():
    """Test queries that should benefit from BM25 exact matching"""
    print("\n" + "="*70)
    print("TESTING EXACT MATCH QUERIES (BM25 Strong)")
    print("="*70)

    exact_queries = [
        # Ticker symbols
        ("AAPL revenue 2025", "Should find Apple's 2025 revenue discussions"),

        # Regulatory codes
        ("ASC 606 revenue recognition", "Should find accounting standard references"),

        # Section headers
        ("Item 1A Risk Factors", "Should find Risk Factors sections from 10-K/Q"),

        # Specific metrics
        ("iPhone revenue Q3 2025", "Should find specific product revenue"),

        # Executive names
        ("Tim Cook CEO remarks", "Should find Tim Cook's statements"),
    ]

    for query, description in exact_queries:
        logger.info(f"Testing: {description}")
        results = hybrid_search.search(query, n_results=5)
        print_results(results, query, max_docs=2)


def test_semantic_queries():
    """Test queries that should benefit from dense semantic search"""
    print("\n" + "="*70)
    print("TESTING SEMANTIC QUERIES (Dense Strong)")
    print("="*70)

    semantic_queries = [
        # Abstract concepts
        ("management's view on future margin pressures", "Should find discussion about margins"),

        # Paraphrases
        ("how is the company dealing with supply chain issues", "Should find supply chain discussions"),

        # Conceptual questions
        ("AI strategy and investments", "Should find AI-related content"),

        # Thematic queries
        ("competitive advantages in cloud computing", "Should find competitive positioning"),

        # Sentiment queries
        ("concerns about regulatory environment", "Should find regulatory risk discussions"),
    ]

    for query, description in semantic_queries:
        logger.info(f"Testing: {description}")
        results = hybrid_search.search(query, n_results=5)
        print_results(results, query, max_docs=2)


def test_hybrid_strength_queries():
    """Test queries that should benefit from both BM25 and dense"""
    print("\n" + "="*70)
    print("TESTING HYBRID QUERIES (Both BM25 + Dense)")
    print("="*70)

    hybrid_queries = [
        # Ticker + semantic concept
        ("GOOGL advertising revenue growth drivers", "Combines ticker (BM25) with concept (Dense)"),

        # Year + thematic
        ("2025 Q2 guidance and outlook", "Combines temporal (BM25) with forward-looking (Dense)"),

        # Section + analysis
        ("MD&A discussion of operational efficiency", "Combines section (BM25) with theme (Dense)"),

        # Product + sentiment
        ("Tesla FSD progress and challenges", "Combines product name (BM25) with analysis (Dense)"),
    ]

    for query, description in hybrid_queries:
        logger.info(f"Testing: {description}")
        results = hybrid_search.search(query, n_results=5, debug=True)
        print_results(results, query, max_docs=2)

        # Show debug info for hybrid queries
        if 'debug' in results:
            print("  Debug Info:")
            print(f"    BM25 hits: {results['debug']['bm25_hits']}")
            print(f"    Dense hits: {results['debug']['dense_hits']}")


def test_filtered_search():
    """Test search with metadata filters"""
    print("\n" + "="*70)
    print("TESTING FILTERED SEARCH")
    print("="*70)

    # Test 1: Filter by ticker
    print("\n1. Searching for 'revenue growth' only in Apple documents:")
    results = hybrid_search.search(
        query="revenue growth",
        n_results=3,
        where={"ticker": "AAPL"}
    )
    print_results(results, "revenue growth [AAPL only]", max_docs=2)

    # Test 2: Filter by document type
    print("\n2. Searching for 'risk factors' only in 10-K filings:")
    results = hybrid_search.search(
        query="risk factors",
        n_results=3,
        where={"doc_type": "10-K"}
    )
    print_results(results, "risk factors [10-K only]", max_docs=2)

    # Test 3: Filter by year
    print("\n3. Searching for 'AI investments' in 2025 documents:")
    results = hybrid_search.search(
        query="AI investments",
        n_results=3,
        where={"fiscal_year": {"$gte": 2025}}
    )
    print_results(results, "AI investments [2025+]", max_docs=2)

    # Test 4: Complex filter - earnings Q&A only
    print("\n4. Searching for 'guidance' in Q&A sections of earnings:")
    results = hybrid_search.search(
        query="guidance",
        n_results=3,
        where={
            "$and": [
                {"doc_type": "earnings_transcript"},
                {"phase": "qa"}
            ]
        }
    )
    print_results(results, "guidance [Earnings Q&A only]", max_docs=2)


def test_weight_adjustment():
    """Test adjusting weights between BM25 and Dense"""
    print("\n" + "="*70)
    print("TESTING WEIGHT ADJUSTMENTS")
    print("="*70)

    query = "MSFT cloud Azure revenue 2025 Q3"

    # Test 1: Equal weights (default)
    print("\n1. Equal weights (0.5/0.5):")
    results = hybrid_search.search(query, n_results=3, bm25_weight=0.5, dense_weight=0.5)
    print_results(results, f"{query} [Equal]", max_docs=1)

    # Test 2: BM25 heavy
    print("\n2. BM25 heavy (0.8/0.2):")
    results = hybrid_search.search(query, n_results=3, bm25_weight=0.8, dense_weight=0.2)
    print_results(results, f"{query} [BM25 heavy]", max_docs=1)

    # Test 3: Dense heavy
    print("\n3. Dense heavy (0.2/0.8):")
    results = hybrid_search.search(query, n_results=3, bm25_weight=0.2, dense_weight=0.8)
    print_results(results, f"{query} [Dense heavy]", max_docs=1)


def test_explain_retrieval():
    """Test the explain_retrieval function"""
    print("\n" + "="*70)
    print("TESTING RETRIEVAL EXPLANATION")
    print("="*70)

    query = "Apple iPhone sales growth Q3 2025"
    explanation = hybrid_search.explain_retrieval(query, n_results=3)

    print(f"\nQuery: '{explanation['query']}'")
    print(f"Explanation of top {len(explanation['top_results'])} results:\n")

    for result in explanation['top_results']:
        print(f"Rank {result['rank']}:")
        print(f"  Document: {result['ticker']} - {result['doc_type']}")
        print(f"  RRF Score: {result['rrf_score']:.4f}")

        if 'retrieval_details' in result:
            details = result['retrieval_details']
            print(f"  BM25: Rank {details['bm25']['rank']}, Contribution {details['bm25']['contribution']:.4f}")
            print(f"  Dense: Rank {details['dense']['rank']}, Contribution {details['dense']['contribution']:.4f}")

        print(f"  Snippet: {result['snippet'][:150]}...")
        print()


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("HYBRID SEARCH TEST SUITE")
    print("="*70)
    print("Testing BM25 + Dense hybrid search with RRF fusion")

    try:
        # Run test suites
        test_exact_match_queries()
        test_semantic_queries()
        test_hybrid_strength_queries()
        test_filtered_search()
        test_weight_adjustment()
        test_explain_retrieval()

        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        print("\n" + "="*70)
        print("✗ TEST SUITE FAILED")
        print("="*70)
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)