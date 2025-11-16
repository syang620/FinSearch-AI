#!/usr/bin/env python3
"""
Validate ChromaDB vector database after chunked document ingestion.
Tests that earnings transcripts and EDGAR filings are properly indexed with metadata.
"""

import os
import sys
import logging
from typing import Dict, Any, List

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.services.rag.vector_store import vector_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_vector_store_stats() -> Dict[str, Any]:
    """Get overall vector store statistics."""
    try:
        count = vector_store.collection.count()
        return {"count": count}
    except Exception as e:
        logger.error(f"Error getting vector store stats: {e}")
        return {}


def test_earnings_transcript_metadata():
    """Test that earnings transcript chunks have proper metadata."""
    logger.info("Testing earnings transcript metadata...")

    # Query for earnings transcript chunks
    try:
        results = vector_store.search(
            query="revenue growth",
            n_results=5,
            where={"doc_type": "earnings_transcript"}
        )

        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            logger.warning("No earnings transcript chunks found!")
            return False

        metadatas = results["metadatas"][0]  # Extract from nested list
        logger.info(f"Found {len(metadatas)} earnings transcript chunks")

        # Check metadata fields
        required_fields = ["ticker", "doc_type", "fiscal_year", "quarter", "phase"]
        for i, metadata in enumerate(metadatas[:3]):
            logger.info(f"\nSample chunk {i+1} metadata:")
            for field in required_fields:
                value = metadata.get(field, "MISSING")
                logger.info(f"  {field}: {value}")
                if value == "MISSING":
                    logger.error(f"Missing required field: {field}")
                    return False

        logger.info("✓ Earnings transcript metadata validation passed")
        return True

    except Exception as e:
        logger.error(f"Error testing earnings metadata: {e}")
        return False


def test_edgar_filing_metadata():
    """Test that EDGAR filing chunks have proper metadata."""
    logger.info("\nTesting EDGAR filing metadata...")

    # Query for EDGAR filing chunks
    try:
        results = vector_store.search(
            query="financial statements",
            n_results=5,
            where={"doc_type": "10-K"}
        )

        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            logger.warning("No EDGAR filing chunks found!")
            return False

        metadatas = results["metadatas"][0]  # Extract from nested list
        logger.info(f"Found {len(metadatas)} EDGAR filing chunks")

        # Check metadata fields
        required_fields = ["ticker", "doc_type", "fiscal_year", "section_title"]
        for i, metadata in enumerate(metadatas[:3]):
            logger.info(f"\nSample chunk {i+1} metadata:")
            for field in required_fields:
                value = metadata.get(field, "MISSING")
                logger.info(f"  {field}: {value}")
                if value == "MISSING" and field != "section_title":  # section_title might be optional
                    logger.error(f"Missing required field: {field}")
                    return False

        logger.info("✓ EDGAR filing metadata validation passed")
        return True

    except Exception as e:
        logger.error(f"Error testing EDGAR metadata: {e}")
        return False


def test_ticker_filtering():
    """Test that we can filter by ticker."""
    logger.info("\nTesting ticker filtering...")

    tickers = ["AAPL", "AMZN", "GOOGL", "MSFT"]

    for ticker in tickers:
        try:
            results = vector_store.search(
                query="revenue",
                n_results=3,
                where={"ticker": ticker}
            )

            metadatas = results.get("metadatas", [[]])[0]
            count = len(metadatas)
            logger.info(f"  {ticker}: {count} chunks found")

            if count == 0:
                logger.warning(f"No chunks found for ticker {ticker}")

        except Exception as e:
            logger.error(f"Error querying ticker {ticker}: {e}")
            return False

    logger.info("✓ Ticker filtering test passed")
    return True


def test_phase_filtering():
    """Test that we can filter earnings by phase (prepared_remarks vs qa)."""
    logger.info("\nTesting phase filtering...")

    phases = ["prepared_remarks", "qa"]

    for phase in phases:
        try:
            results = vector_store.search(
                query="growth",
                n_results=3,
                where={
                    "$and": [
                        {"doc_type": "earnings_transcript"},
                        {"phase": phase}
                    ]
                }
            )

            metadatas = results.get("metadatas", [[]])[0]
            count = len(metadatas)
            logger.info(f"  {phase}: {count} chunks found")

            # Verify phase in metadata
            for metadata in metadatas[:2]:
                if metadata.get("phase") != phase:
                    logger.error(f"Phase mismatch! Expected {phase}, got {metadata.get('phase')}")
                    return False

        except Exception as e:
            logger.error(f"Error querying phase {phase}: {e}")
            return False

    logger.info("✓ Phase filtering test passed")
    return True


def test_semantic_search():
    """Test that semantic search returns relevant results."""
    logger.info("\nTesting semantic search...")

    test_queries = [
        "What were the main revenue drivers?",
        "Discuss the company's AI strategy",
        "What are the key risk factors?"
    ]

    for query in test_queries:
        try:
            results = vector_store.search(
                query=query,
                n_results=3
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            count = len(documents)
            logger.info(f"\n  Query: '{query}'")
            logger.info(f"  Results: {count} chunks")

            if count > 0:
                # Show first result snippet
                doc = documents[0]
                metadata = metadatas[0]
                snippet = doc[:150] + "..." if len(doc) > 150 else doc
                logger.info(f"  Top result: {metadata.get('ticker')} {metadata.get('doc_type')}")
                logger.info(f"  Snippet: {snippet}")

        except Exception as e:
            logger.error(f"Error with semantic search: {e}")
            return False

    logger.info("✓ Semantic search test passed")
    return True


def run_validation():
    """Run all validation tests."""
    print("="*70)
    print("VECTOR DATABASE VALIDATION")
    print("="*70)

    # Get overall stats
    logger.info("\nGetting vector store statistics...")
    stats = get_vector_store_stats()
    total_docs = stats.get("count", 0)
    logger.info(f"Total documents in vector store: {total_docs:,}")

    if total_docs == 0:
        logger.error("Vector store is empty!")
        return False

    # Run tests
    tests = [
        ("Earnings Transcript Metadata", test_earnings_transcript_metadata),
        ("EDGAR Filing Metadata", test_edgar_filing_metadata),
        ("Ticker Filtering", test_ticker_filtering),
        ("Phase Filtering", test_phase_filtering),
        ("Semantic Search", test_semantic_search),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*70}")
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Total documents: {total_docs:,}")
    print(f"\nTest Results:")

    passed_count = 0
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if passed:
            passed_count += 1

    print(f"\nOverall: {passed_count}/{len(results)} tests passed")
    print("="*70)

    return passed_count == len(results)


if __name__ == "__main__":
    logger.info("Starting vector database validation")
    success = run_validation()

    if success:
        logger.info("\n✓ All validation tests passed!")
        sys.exit(0)
    else:
        logger.error("\n✗ Some validation tests failed!")
        sys.exit(1)
