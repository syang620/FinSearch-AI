#!/usr/bin/env python3
"""
Validation script for BM25+Dense hybrid search system
Ensures both indices are properly populated and hybrid search performs well
"""

import sys
import os
import logging
from typing import Dict, Any, List, Tuple

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.services.rag.vector_store import vector_store
from app.services.rag.bm25_index import bm25_index
from app.services.rag.hybrid_search import hybrid_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_indices() -> Tuple[bool, Dict[str, Any]]:
    """Validate that both BM25 and Dense indices are populated"""
    stats = {}

    # Check ChromaDB (Dense)
    try:
        dense_count = vector_store.collection.count()
        stats['dense_count'] = dense_count
        logger.info(f"Dense index (ChromaDB): {dense_count:,} documents")
    except Exception as e:
        logger.error(f"Error accessing dense index: {e}")
        stats['dense_count'] = 0
        return False, stats

    # Check BM25 (SQLite FTS5)
    try:
        bm25_count = bm25_index.count()
        stats['bm25_count'] = bm25_count
        logger.info(f"BM25 index (SQLite): {bm25_count:,} documents")
    except Exception as e:
        logger.error(f"Error accessing BM25 index: {e}")
        stats['bm25_count'] = 0
        return False, stats

    # Both should have documents
    if dense_count == 0 or bm25_count == 0:
        logger.error("One or both indices are empty!")
        return False, stats

    # Counts should be similar (within 10% tolerance)
    diff_pct = abs(dense_count - bm25_count) / max(dense_count, bm25_count)
    stats['diff_percentage'] = diff_pct * 100

    if diff_pct > 0.1:
        logger.warning(f"Document count mismatch: Dense={dense_count}, BM25={bm25_count} ({diff_pct*100:.1f}% difference)")
    else:
        logger.info(f"✓ Document counts aligned (difference: {diff_pct*100:.1f}%)")

    return True, stats


def test_query_types() -> Dict[str, List[Dict]]:
    """Test different query types to validate retrieval quality"""
    test_results = {}

    test_cases = [
        # Exact match queries (BM25 strong)
        {
            'category': 'exact_match',
            'queries': [
                ('AAPL', "Ticker symbol search"),
                ('10-K', "Document type search"),
                ('Item 1A', "Section header search"),
                ('ASC 606', "Accounting standard search"),
                ('Q3 2025', "Time period search")
            ]
        },
        # Semantic queries (Dense strong)
        {
            'category': 'semantic',
            'queries': [
                ('revenue growth drivers', "Conceptual search"),
                ('margin improvement strategies', "Strategy search"),
                ('competitive positioning', "Analysis search"),
                ('future outlook', "Forward-looking search"),
                ('operational challenges', "Risk search")
            ]
        },
        # Hybrid queries (Both needed)
        {
            'category': 'hybrid',
            'queries': [
                ('Apple iPhone revenue Q3 2025', "Ticker + product + time"),
                ('Microsoft cloud Azure growth', "Company + product + metric"),
                ('Tesla FSD progress challenges', "Company + product + analysis"),
                ('Google advertising revenue trends', "Company + segment + pattern"),
                ('Meta AI investments 2025', "Company + technology + year")
            ]
        }
    ]

    for test_case in test_cases:
        category = test_case['category']
        results = []

        logger.info(f"\nTesting {category} queries...")

        for query, description in test_case['queries']:
            try:
                # Run hybrid search
                search_results = hybrid_search.search(query, n_results=5, debug=True)

                # Analyze results
                num_results = len(search_results['ids'][0]) if search_results['ids'] else 0

                result_info = {
                    'query': query,
                    'description': description,
                    'num_results': num_results,
                    'success': num_results > 0
                }

                if 'debug' in search_results:
                    result_info['bm25_hits'] = search_results['debug']['bm25_hits']
                    result_info['dense_hits'] = search_results['debug']['dense_hits']

                if num_results > 0:
                    # Get top result metadata
                    top_meta = search_results['metadatas'][0][0]
                    result_info['top_ticker'] = top_meta.get('ticker', 'N/A')
                    result_info['top_doc_type'] = top_meta.get('doc_type', 'N/A')

                results.append(result_info)

                if num_results == 0:
                    logger.warning(f"  ✗ '{query}': No results found")
                else:
                    logger.info(f"  ✓ '{query}': {num_results} results (BM25: {result_info.get('bm25_hits', 0)}, Dense: {result_info.get('dense_hits', 0)})")

            except Exception as e:
                logger.error(f"  ✗ '{query}': Error - {e}")
                results.append({
                    'query': query,
                    'description': description,
                    'success': False,
                    'error': str(e)
                })

        test_results[category] = results

    return test_results


def test_filtering() -> Dict[str, bool]:
    """Test metadata filtering capabilities"""
    filter_tests = {}

    logger.info("\nTesting metadata filtering...")

    # Test ticker filter
    try:
        results = hybrid_search.search("revenue", n_results=3, where={"ticker": "MSFT"})
        count = len(results['ids'][0]) if results['ids'] else 0
        all_msft = all(
            m.get('ticker') == 'MSFT'
            for m in results.get('metadatas', [[]])[0]
        ) if count > 0 else False

        filter_tests['ticker_filter'] = count > 0 and all_msft
        logger.info(f"  Ticker filter (MSFT): {'✓ PASS' if filter_tests['ticker_filter'] else '✗ FAIL'} ({count} results)")
    except Exception as e:
        filter_tests['ticker_filter'] = False
        logger.error(f"  Ticker filter: ✗ FAIL - {e}")

    # Test document type filter
    try:
        results = hybrid_search.search("risk", n_results=3, where={"doc_type": "10-K"})
        count = len(results['ids'][0]) if results['ids'] else 0
        all_10k = all(
            m.get('doc_type') == '10-K'
            for m in results.get('metadatas', [[]])[0]
        ) if count > 0 else False

        filter_tests['doc_type_filter'] = count > 0 and all_10k
        logger.info(f"  Doc type filter (10-K): {'✓ PASS' if filter_tests['doc_type_filter'] else '✗ FAIL'} ({count} results)")
    except Exception as e:
        filter_tests['doc_type_filter'] = False
        logger.error(f"  Doc type filter: ✗ FAIL - {e}")

    # Test year filter
    try:
        results = hybrid_search.search("guidance", n_results=3, where={"fiscal_year": {"$gte": 2025}})
        count = len(results['ids'][0]) if results['ids'] else 0
        all_2025_plus = all(
            m.get('fiscal_year', 0) >= 2025
            for m in results.get('metadatas', [[]])[0]
        ) if count > 0 else False

        filter_tests['year_filter'] = count > 0 and all_2025_plus
        logger.info(f"  Year filter (>=2025): {'✓ PASS' if filter_tests['year_filter'] else '✗ FAIL'} ({count} results)")
    except Exception as e:
        filter_tests['year_filter'] = False
        logger.error(f"  Year filter: ✗ FAIL - {e}")

    # Test combined filter (earnings Q&A)
    try:
        results = hybrid_search.search(
            "outlook",
            n_results=3,
            where={
                "$and": [
                    {"doc_type": "earnings_transcript"},
                    {"phase": "qa"}
                ]
            }
        )
        count = len(results['ids'][0]) if results['ids'] else 0
        all_qa = all(
            m.get('doc_type') == 'earnings_transcript' and m.get('phase') == 'qa'
            for m in results.get('metadatas', [[]])[0]
        ) if count > 0 else False

        filter_tests['combined_filter'] = count > 0 and all_qa
        logger.info(f"  Combined filter (Earnings Q&A): {'✓ PASS' if filter_tests['combined_filter'] else '✗ FAIL'} ({count} results)")
    except Exception as e:
        filter_tests['combined_filter'] = False
        logger.error(f"  Combined filter: ✗ FAIL - {e}")

    return filter_tests


def test_rrf_fusion() -> Dict[str, Any]:
    """Test that RRF fusion is working correctly"""
    logger.info("\nTesting RRF fusion mechanics...")

    query = "Apple revenue growth 2025"

    # Get results with different weight configurations
    configs = [
        {'bm25_weight': 0.5, 'dense_weight': 0.5, 'name': 'Balanced'},
        {'bm25_weight': 0.8, 'dense_weight': 0.2, 'name': 'BM25-heavy'},
        {'bm25_weight': 0.2, 'dense_weight': 0.8, 'name': 'Dense-heavy'},
    ]

    results_comparison = {}

    for config in configs:
        try:
            results = hybrid_search.search(
                query,
                n_results=5,
                bm25_weight=config['bm25_weight'],
                dense_weight=config['dense_weight'],
                debug=True
            )

            top_ids = results['ids'][0][:3] if results['ids'] else []

            results_comparison[config['name']] = {
                'top_3_ids': top_ids,
                'bm25_hits': results.get('debug', {}).get('bm25_hits', 0),
                'dense_hits': results.get('debug', {}).get('dense_hits', 0),
                'weights': f"BM25={config['bm25_weight']}, Dense={config['dense_weight']}"
            }

            logger.info(f"  {config['name']}: {len(top_ids)} results retrieved")

        except Exception as e:
            logger.error(f"  {config['name']}: Error - {e}")
            results_comparison[config['name']] = {'error': str(e)}

    # Check if different weights produce different rankings
    if len(results_comparison) >= 2:
        rankings = [v.get('top_3_ids', []) for v in results_comparison.values()]
        all_same = all(r == rankings[0] for r in rankings[1:])

        if all_same:
            logger.warning("  ⚠ Different weights produced identical rankings")
        else:
            logger.info("  ✓ Different weights produce different rankings (RRF working)")

    return results_comparison


def run_full_validation():
    """Run complete validation suite"""
    print("\n" + "="*70)
    print("HYBRID SEARCH SYSTEM VALIDATION")
    print("="*70)

    all_passed = True
    validation_results = {}

    # 1. Validate indices
    print("\n1. INDEX VALIDATION")
    print("-" * 40)
    indices_valid, index_stats = validate_indices()
    validation_results['indices'] = {
        'valid': indices_valid,
        'stats': index_stats
    }
    all_passed = all_passed and indices_valid

    # 2. Test query types
    print("\n2. QUERY TYPE TESTING")
    print("-" * 40)
    query_results = test_query_types()
    validation_results['queries'] = query_results

    # Calculate success rates
    for category, results in query_results.items():
        success_rate = sum(1 for r in results if r.get('success', False)) / len(results) * 100
        print(f"  {category}: {success_rate:.0f}% success rate")
        if success_rate < 80:
            all_passed = False

    # 3. Test filtering
    print("\n3. METADATA FILTERING")
    print("-" * 40)
    filter_results = test_filtering()
    validation_results['filtering'] = filter_results

    filter_pass_rate = sum(1 for v in filter_results.values() if v) / len(filter_results) * 100
    print(f"  Overall: {filter_pass_rate:.0f}% of filters working")
    if filter_pass_rate < 75:
        all_passed = False

    # 4. Test RRF fusion
    print("\n4. RRF FUSION VALIDATION")
    print("-" * 40)
    rrf_results = test_rrf_fusion()
    validation_results['rrf'] = rrf_results

    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    print(f"\nIndex Statistics:")
    print(f"  Dense documents: {index_stats.get('dense_count', 0):,}")
    print(f"  BM25 documents: {index_stats.get('bm25_count', 0):,}")
    print(f"  Difference: {index_stats.get('diff_percentage', 0):.1f}%")

    print(f"\nQuery Performance:")
    for category, results in query_results.items():
        success_count = sum(1 for r in results if r.get('success', False))
        print(f"  {category}: {success_count}/{len(results)} queries successful")

    print(f"\nFilter Tests:")
    for filter_name, passed in filter_results.items():
        print(f"  {filter_name}: {'✓ PASS' if passed else '✗ FAIL'}")

    print(f"\nRRF Configurations Tested: {len(rrf_results)}")

    print("\n" + "="*70)
    if all_passed:
        print("✓ HYBRID SEARCH VALIDATION PASSED")
    else:
        print("✗ SOME VALIDATION TESTS FAILED")
    print("="*70)

    return all_passed, validation_results


if __name__ == "__main__":
    success, results = run_full_validation()
    sys.exit(0 if success else 1)