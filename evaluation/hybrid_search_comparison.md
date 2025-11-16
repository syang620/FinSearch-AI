# Hybrid Search Integration - Evaluation Results

## Summary

We successfully integrated hybrid search (BM25 + Dense Vector) into the FinSearch AI retrieval system and compared its performance against vector-only search.

## Implementation Changes

1. **Fixed BM25 Query Handling**
   - Fixed apostrophe handling in queries (e.g., "Apple's" → "Apple s")
   - Fixed comma handling to prevent FTS5 syntax errors
   - Wrapped each word in quotes for proper FTS5 processing

2. **Fixed Filter Field Mismatch**
   - Mapped `company` field (used by vector store) to `ticker` field (used by BM25)
   - Ensured consistent filtering across both retrieval methods

3. **Updated Retriever**
   - Changed from vector-only search to hybrid search
   - Set equal weights (0.5 each) for BM25 and dense vector search
   - Uses Reciprocal Rank Fusion (RRF) with k=60 for result merging

## Performance Comparison

### Vector-Only Search (Baseline)
- **Hit Rate@5**: 60.7%
- **MRR**: 0.361
- **Hit Rate@1**: 21.4%
- **Hit Rate@3**: 46.4%
- **Answer Coverage**: 28.6%
- **Avg Latency**: 300ms

### Hybrid Search (BM25 + Dense) - WORKING
- **Hit Rate@5**: 60.7% (same)
- **MRR**: 0.325 (-10% decrease)
- **Hit Rate@1**: 17.9% (-16% decrease)
- **Hit Rate@3**: 42.9% (-7.5% decrease)
- **Answer Coverage**: 28.6% (same)
- **Avg Latency**: 224ms (25% faster)

## Analysis

### Key Findings with Working Hybrid Search

1. **Slight Performance Degradation**
   - MRR decreased by 10% (0.361 → 0.325)
   - Hit Rate@1 decreased by 16% (21.4% → 17.9%)
   - Hit Rate@3 decreased by 7.5% (46.4% → 42.9%)
   - Hit Rate@5 remained the same at 60.7%

2. **Latency Improvement**
   - Average latency improved by 25% (300ms → 224ms)
   - This is likely due to better caching and parallel search execution

3. **Why Did Performance Slightly Decrease?**
   - **Dilution Effect**: BM25 may be introducing less relevant results based on keyword matches that aren't semantically relevant
   - **Equal Weighting**: The 50/50 weight split may not be optimal for financial queries
   - **Term Mismatch**: Financial documents use varied terminology (revenue vs sales, income vs earnings)
   - **RRF Bias**: Documents appearing in only one search method get lower ranks in fusion

### Category Performance Comparison

| Category | Questions | Vector-Only MRR | Hybrid MRR | Change |
|----------|-----------|-----------------|------------|--------|
| single_doc_fact | 10 | 0.208 | 0.123 | -40.9% |
| single_doc_context | 8 | 0.410 | 0.410 | 0% |
| single_doc_multi_period | 6 | 0.208 | 0.181 | -13.0% |
| cross_doc_fact | 4 | 0.875 | 0.875 | 0% |

**Key Observations:**
- Single document fact queries suffered the most (-40.9% MRR)
- Context and cross-document queries remained stable
- BM25 appears to be diluting results for precise fact retrieval

## Recommendations

1. **Adjust Weights Based on Query Type**
   - For fact queries: Consider 0.2 BM25 / 0.8 Dense (since BM25 is hurting fact retrieval)
   - For context queries: Keep 0.5/0.5 (working well)
   - For cross-document: Keep current weights (excellent performance)

2. **Query Expansion for BM25**
   - Add synonyms for financial terms (revenue ↔ sales, income ↔ earnings)
   - Preprocess queries to expand abbreviations (Q1 → "first quarter", FY → "fiscal year")

3. **Optimize RRF Parameter**
   - Current k=60 may be too high
   - Try k=30 for stronger emphasis on top-ranked documents

4. **Consider Adaptive Hybrid Strategy**
   - Use vector-only for numeric/fact queries
   - Use hybrid for context/narrative queries

## Conclusion

The working hybrid search system shows mixed results:

**Positives:**
- ✅ 25% faster latency (224ms vs 300ms)
- ✅ Maintains same Hit@5 rate (60.7%)
- ✅ Excellent on cross-document queries (MRR 0.875)

**Negatives:**
- ❌ 10% lower MRR overall (0.325 vs 0.361)
- ❌ 40.9% worse on single-doc fact queries
- ❌ Lower precision at top ranks (Hit@1 and Hit@3)

**Bottom Line:** For this financial Q&A use case, the dense vector embeddings (BGE) alone are performing better than the hybrid approach with equal weights. The BM25 component appears to be introducing noise rather than improving retrieval quality, particularly for precise fact-based queries. Consider using vector-only search or heavily weighting toward dense search (e.g., 0.8 dense / 0.2 BM25) for better results.