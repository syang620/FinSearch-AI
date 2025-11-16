# Detailed Evaluation Review - With LLM Reranker

Generated: 2025-11-16
Total Questions: 28
System Configuration: Hybrid Search + LLM Reranker (qwen2.5:0.5b)

## Summary Metrics

| Metric | Value | Status |
|--------|-------|---------|
| Hit Rate@1 | 46.4% | ✅ Improved from 17.9% |
| Hit Rate@3 | 75.0% | ✅ Improved from 42.9% |
| Hit Rate@5 | 78.6% | ✅ Improved from 60.7% |
| MRR | 0.610 | ✅ Improved from 0.325 |
| Answer Coverage | 35.7% | - |
| Avg Latency | 9738ms | ⚠️ Increased from 224ms |

---

## Single Document Fact Questions

### 1. ✅ AAPL_fact_rev_2024

**Question**: What were Apple's total net sales in fiscal 2024?

**Expected Answer**: 391035 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-K_2025.htm ✓
  3. raw_10-Q_2024_Q1.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: Yes
  - Latency: 12317ms

**Analysis**: Reranker successfully promoted 10-K_2025 to position 2 (was not in top 3 before)

---

### 2. ✅ AAPL_fact_rev_2025

**Question**: What were Apple's total net sales in fiscal 2025?

**Expected Answer**: 416161 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-K_2025.htm ✓
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: Yes
  - Latency: 10158ms

**Analysis**: Reranker correctly prioritized 10-K over 10-Q for annual query

---

### 3. ❌ AAPL_fact_netincome_2024

**Question**: What was Apple's net income in fiscal 2024?

**Expected Answer**: 93736 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-Q_2024_Q1.htm
  3. raw_10-Q_2025_Q1.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 10543ms

**Analysis**: Expected source not retrieved, but at least 10-K ranked first over 10-Q

---

### 4. ⚠️ AAPL_fact_netincome_2025

**Question**: What was Apple's net income in fiscal 2025?

**Expected Answer**: 112010 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-Q_2025_Q3.htm
  3. raw_10-Q_2025_Q1.htm
  4. (Not shown)
  5. raw_10-K_2025.htm ✓

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.250
  - Answer in Retrieved Text: No
  - Latency: 9476ms

**Analysis**: Found in position 5, reranker needs tuning for net income queries

---

### 5. ✅ AAPL_fact_servicesrev_2025

**Question**: In fiscal 2025, what were Apple's Services net sales?

**Expected Answer**: 109158 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm ✓
  2. raw_10-K_2024.htm
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 9720ms

**Analysis**: Perfect retrieval - correct document at position 1

---

### 6. ✅ AAPL_fact_iphonerev_2025

**Question**: In fiscal 2025, what were Apple's iPhone net sales?

**Expected Answer**: 204498 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm ✓
  2. raw_10-K_2025.htm ✓
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 10644ms

**Analysis**: Perfect retrieval with duplicate correct results

---

### 7. ❌ AAPL_fact_americasrev_2024

**Question**: What was Apple's Americas net sales in fiscal 2024?

**Expected Answer**: 162560 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. Unknown
  2. Unknown
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: Yes
  - Latency: 9346ms

**Analysis**: Geographic data retrieval issue, many "Unknown" sources

---

### 8. ❌ AAPL_fact_greaterchinarev_2025

**Question**: What was Apple's Greater China net sales in fiscal 2025?

**Expected Answer**: 60307 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q2.htm
  2. raw_10-Q_2024_Q1.htm
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 9872ms

**Analysis**: All 10-Q documents retrieved, 10-K not found for geographic query

---

### 9. ✅ AAPL_fact_distribution_split_2024

**Question**: What percentage of Apple's net sales came from direct vs indirect distribution channels in fiscal 2024?

**Expected Answer**: Direct: 40%, Indirect: 60%

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm ✓
  2. raw_10-K_2024.htm ✓
  3. raw_10-K_2025.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 10001ms

**Analysis**: Perfect retrieval with correct document at top positions

---

### 10. ✅ AAPL_fact_distribution_split_2025

**Question**: What percentage of Apple's net sales came from direct vs indirect distribution channels in fiscal 2025?

**Expected Answer**: Direct: 40%, Indirect: 60%

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm ✓
  2. raw_10-K_2024.htm
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 8985ms

**Analysis**: Perfect retrieval - correct annual report prioritized

---

## Single Document Context Questions

### 11. ✅ AAPL_context_products_2024

**Question**: What are Apple's main product categories and their descriptions?

**Expected Answer**: [Product descriptions from 10-K]

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm ✓
  2. raw_10-K_2024.htm ✓
  3. raw_10-K_2024.htm ✓

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 8588ms

**Analysis**: Perfect retrieval with all results from correct document

---

### 12. ✅ AAPL_context_wearables_2024

**Question**: What products are included in Apple's Wearables, Home and Accessories category?

**Expected Answer**: AirPods, Apple TV, Apple Watch, HomePod, and accessories

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm ✓
  2. raw_10-K_2024.htm ✓
  3. raw_10-K_2024.htm ✓

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 7619ms

**Analysis**: Perfect retrieval, fastest query response

---

### 13. ✅ AAPL_context_segments_2024

**Question**: How does Apple organize its business segments?

**Expected Answer**: Single reportable segment

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm ✓
  2. raw_10-K_2025.htm
  3. raw_10-Q_2024_Q1.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 9656ms

**Analysis**: Correct document at position 1

---

### 14. ✅ AAPL_context_markets_2024

**Question**: What geographic markets does Apple operate in and how are they defined?

**Expected Answer**: Americas, Europe, Greater China, Japan, Rest of Asia Pacific

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm ✓
  2. raw_10-K_2024.htm ✓
  3. raw_10-Q_2025_Q1.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 8714ms

**Analysis**: Perfect retrieval with duplicate correct results

---

### 15. ✅ AAPL_context_competitivefactors_2025

**Question**: What competitive factors affect Apple's business?

**Expected Answer**: Price, product features, quality, design innovation, etc.

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-K_2025.htm ✓
  3. raw_10-K_2025.htm ✓

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: Yes
  - Latency: 8130ms

**Analysis**: Correct document at position 2, 10-Q incorrectly ranked first

---

### 16. ✅ AAPL_context_competition_2025

**Question**: How does Apple characterize the competitive conditions in its markets?

**Expected Answer**: Highly competitive, rapid technological advances

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-K_2025.htm ✓
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: No
  - Latency: 9439ms

**Analysis**: Correct document at position 2

---

### 17. ✅ AAPL_context_humancapital_2024

**Question**: What information does Apple provide about its human capital and employees?

**Expected Answer**: Employee count, diversity initiatives, compensation philosophy

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm
  2. raw_10-K_2024.htm ✓
  3. raw_10-K_2024.htm ✓

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: No
  - Latency: 10292ms

**Analysis**: Correct document at positions 2-3, wrong year at position 1

---

### 18. ✅ AAPL_context_humancapital_2025

**Question**: What does Apple disclose about employee information in its 2025 10-K?

**Expected Answer**: Full-time equivalent employees count, etc.

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm ✓
  2. raw_10-K_2025.htm ✓
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 9658ms

**Analysis**: Perfect retrieval with correct document at top

---

## Multi-Period Questions

### 19. ✅ AAPL_multi_rev_years

**Question**: What were Apple's total net sales for fiscal years 2023, 2024, and 2025?

**Expected Answer**: 2023: 383285M, 2024: 391035M, 2025: 416161M

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-K_2025.htm ✓
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: No
  - Latency: 10437ms

**Analysis**: Correct document at position 2, reranker working

---

### 20. ✅ AAPL_multi_iphonerev_years

**Question**: What were Apple's iPhone net sales for fiscal years 2023, 2024, and 2025?

**Expected Answer**: 2023: 200583M, 2024: 201183M, 2025: 204498M

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm ✓
  2. raw_10-K_2024.htm
  3. raw_10-K_2025.htm ✓

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 9064ms

**Analysis**: Perfect retrieval for multi-year iPhone data

---

### 21. ✅ AAPL_multi_servicesrev_years

**Question**: What were Apple's Services net sales for fiscal years 2023, 2024, and 2025?

**Expected Answer**: 2023: 85200M, 2024: 96169M, 2025: 109158M

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-K_2025.htm ✓
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: No
  - Latency: 9266ms

**Analysis**: Correct document at position 2

---

### 22. ❌ AAPL_multi_americasrev_years

**Question**: What were Apple's Americas net sales for fiscal years 2023, 2024, and 2025?

**Expected Answer**: 2023: 162560M, 2024: 167040M, 2025: 178529M

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q3.htm
  2. raw_10-Q_2024_Q3.htm
  3. Unknown

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 10398ms

**Analysis**: Geographic multi-year queries failing, only quarterly docs retrieved

---

### 23. ❌ AAPL_multi_grchinarev_years

**Question**: What were Apple's Greater China net sales for fiscal years 2023, 2024, and 2025?

**Expected Answer**: 2023: 72559M, 2024: 66956M, 2025: 60307M

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2024_Q2.htm
  2. raw_10-Q_2025_Q1.htm
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 10947ms

**Analysis**: All quarterly documents, 10-K not retrieved for geographic data

---

### 24. ✅ AAPL_multi_grossmargin_2025vs2024

**Question**: How did Apple's gross margin change between fiscal 2024 and 2025?

**Expected Answer**: 2024: 45.59%, 2025: 48.41%

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-Q_2025_Q3.htm
  3. raw_10-K_2025.htm ✓

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.333
  - Answer in Retrieved Text: No
  - Latency: 9648ms

**Analysis**: Correct document at position 3, could be ranked higher

---

## Cross-Document Questions

### 25. ✅ AAPL_cross_q1_rev_2024_2025

**Question**: Compare Apple's Q1 revenue between fiscal 2024 and 2025

**Expected Answer**: Q1 2024: 119575M, Q1 2025: 124300M

**Expected Source(s)**: raw_10-Q_2024_Q1.htm, raw_10-Q_2025_Q1.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm ✓
  2. raw_10-Q_2024_Q1.htm ✓
  3. raw_10-Q_2024_Q1.htm ✓

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 10251ms

**Analysis**: Perfect retrieval of both required quarterly documents

---

### 26. ✅ AAPL_cross_q2_rev_2024_2025

**Question**: Compare Apple's Q2 revenue between fiscal 2024 and 2025

**Expected Answer**: Q2 2024: 90753M, Q2 2025: 94836M

**Expected Source(s)**: raw_10-Q_2024_Q2.htm, raw_10-Q_2025_Q2.htm

**Retrieved Sources**:
  1. raw_10-Q_2024_Q2.htm ✓
  2. raw_10-Q_2025_Q2.htm ✓
  3. raw_10-Q_2025_Q1.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 9641ms

**Analysis**: Perfect retrieval of both required documents

---

### 27. ✅ AAPL_cross_q3_rev_2024_2025

**Question**: Compare Apple's Q3 revenue between fiscal 2024 and 2025

**Expected Answer**: Q3 2024: 85777M, Q3 2025: 85468M

**Expected Source(s)**: raw_10-Q_2024_Q3.htm, raw_10-Q_2025_Q3.htm

**Retrieved Sources**:
  1. raw_10-Q_2024_Q3.htm ✓
  2. raw_10-Q_2025_Q1.htm
  3. raw_10-Q_2025_Q3.htm ✓

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 9061ms

**Analysis**: Both required documents retrieved, one at position 1, one at position 3

---

### 28. ❌ AAPL_cross_q4_rev_2025_year_2025

**Question**: How does Apple's Q4 2025 revenue compare to full fiscal year 2025?

**Expected Answer**: Q4 2025: 94930M, FY 2025: 416161M

**Expected Source(s)**: raw_10-Q_2024_Q4.htm, raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2024_Q1.htm
  2. raw_10-Q_2025_Q3.htm
  3. raw_10-Q_2025_Q1.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 10801ms

**Analysis**: Neither required document retrieved, Q4 data retrieval issue

---

## Key Observations

### Successes with Reranker
1. **10-K vs 10-Q distinction working**: Annual queries now correctly prioritize 10-K documents
2. **Perfect retrieval for product/service queries**: All context questions about products achieving MRR 1.0
3. **Cross-document quarterly comparisons**: Q1-Q3 comparisons working perfectly

### Remaining Issues
1. **Geographic data queries failing**: Americas and Greater China queries not finding 10-K documents
2. **Q4 data retrieval problems**: Q4 2025 data not being retrieved
3. **Net income queries need improvement**: Lower MRR than revenue queries

### Performance Trade-offs
- Average latency increased from ~224ms to ~9738ms
- Hit Rate@5 improved from 60.7% to 78.6%
- MRR nearly doubled from 0.325 to 0.610

The reranker has successfully addressed the main issue of document type mismatches while maintaining reasonable performance for a financial research system.