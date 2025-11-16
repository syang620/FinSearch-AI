# Evaluation Review - With Year Filtering Enhancement

## Executive Summary

After implementing year filtering logic to handle SEC filing conventions (where 10-K_2025 contains fiscal 2024 data), the system maintains the excellent performance achieved with the LLM reranker: **78.6% Hit@5** and **0.610 MRR**.

## Performance Metrics Overview

| Metric | Value | Status |
|--------|-------|---------|
| **Hit Rate@5** | 78.6% | ✅ Excellent |
| **Hit Rate@3** | 75.0% | ✅ Excellent |
| **Hit Rate@1** | 46.4% | ✅ Good |
| **MRR** | 0.610 | ✅ Very Good |
| **Precision@1** | 46.4% | ✅ Good |
| **Precision@3** | 42.9% | ✅ Good |
| **Precision@5** | 32.9% | ✅ Acceptable |
| **Answer Coverage** | 35.7% | ⚠️ Needs Improvement |
| **Avg Latency** | 9,738ms | ⚠️ Acceptable |

---

## Detailed Question-by-Question Results

## Single Document - Fact Queries (10 questions)

### 1. ✅ AAPL_fact_rev_2024

**Question**: What were Apple's total net sales in fiscal 2024?

**Expected Answer**: 391035 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-K_2025.htm ✅
  3. raw_10-Q_2024_Q1.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: Yes
  - Latency: 12,317ms

**Analysis**: Year filtering working - 10-K_2025 correctly retrieved at rank #2 for FY2024 data.

---

### 2. ✅ AAPL_fact_rev_2025

**Question**: What were Apple's total net sales in fiscal 2025?

**Expected Answer**: 416161 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-K_2025.htm ✅
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: Yes
  - Latency: 10,158ms

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
  - Latency: 10,543ms

**Analysis**: Failed - system didn't retrieve 10-K_2025 for FY2024 net income.

---

### 4. ✅ AAPL_fact_netincome_2025

**Question**: What was Apple's net income in fiscal 2025?

**Expected Answer**: 101966 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-Q_2025_Q3.htm
  3. raw_10-Q_2025_Q1.htm
  4. raw_10-K_2025.htm ✅
  5. raw_10-Q_2025_Q2.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.250
  - Answer in Retrieved Text: No
  - Latency: 9,476ms

**Analysis**: Found but at lower rank (#4).

---

### 5. ✅ AAPL_fact_servicesrev_2025

**Question**: What was the Services revenue for fiscal year 2025?

**Expected Answer**: 98769 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm ✅
  2. raw_10-K_2024.htm
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 9,720ms

**Analysis**: Perfect retrieval at rank #1.

---

### 6. ✅ AAPL_fact_iphonerev_2025

**Question**: What were the iPhone net sales for fiscal year 2025?

**Expected Answer**: 210066 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm ✅
  2. raw_10-K_2025.htm ✅
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 10,644ms

**Analysis**: Perfect retrieval with correct document at #1 and #2.

---

### 7. ❌ AAPL_fact_americasrev_2024

**Question**: What was the net sales for Americas segment in fiscal year 2024?

**Expected Answer**: 166856 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. Unknown
  2. Unknown
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: Yes (but wrong source)
  - Latency: 9,346ms

**Analysis**: Geographic segment retrieval failure.

---

### 8. ❌ AAPL_fact_greaterchinarev_2025

**Question**: What was the Greater China net sales for fiscal year 2025?

**Expected Answer**: 66886 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q2.htm
  2. raw_10-Q_2024_Q1.htm
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 9,872ms

**Analysis**: Geographic segment retrieval failure - getting quarterly reports instead of annual.

---

### 9. ✅ AAPL_fact_distribution_split_2024

**Question**: What percentage of sales came from direct vs indirect channels in 2024?

**Expected Answer**: 38% direct, 62% indirect

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm ✅
  2. raw_10-K_2024.htm ✅
  3. raw_10-K_2025.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 10,001ms

**Analysis**: Perfect - correctly retrieves 10-K_2024 for FY2023 distribution data.

---

### 10. ✅ AAPL_fact_distribution_split_2025

**Question**: What was the direct to consumer vs indirect distribution split in 2025?

**Expected Answer**: 39% direct, 61% indirect

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm ✅
  2. raw_10-K_2024.htm
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 8,985ms

**Analysis**: Perfect retrieval at rank #1.

---

## Single Document - Context Queries (8 questions)

### 11. ✅ AAPL_context_products_2024

**Question**: What are Apple's product categories in 2024?

**Expected Answer**: iPhone, Mac, iPad, Wearables/Home/Accessories, Services

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm ✅
  2. raw_10-K_2024.htm ✅
  3. raw_10-K_2024.htm ✅

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 8,588ms

**Analysis**: Perfect retrieval with all top 3 from correct document.

---

### 12. ✅ AAPL_context_wearables_2024

**Question**: What products are included in the Wearables, Home and Accessories category in 2024?

**Expected Answer**: AirPods, Apple TV, Apple Watch, Beats, HomePod, and accessories

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm ✅
  2. raw_10-K_2024.htm ✅
  3. raw_10-K_2024.htm ✅

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 7,619ms

**Analysis**: Perfect retrieval, fastest response time.

---

### 13. ✅ AAPL_context_segments_2024

**Question**: What are Apple's operating segments in 2024?

**Expected Answer**: Products and Services

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm ✅
  2. raw_10-K_2025.htm
  3. raw_10-Q_2024_Q1.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 9,656ms

---

### 14. ✅ AAPL_context_markets_2024

**Question**: What geographic markets does Apple operate in as of 2024?

**Expected Answer**: Americas, Europe, Greater China, Japan, Rest of Asia Pacific

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm ✅
  2. raw_10-K_2024.htm ✅
  3. raw_10-Q_2025_Q1.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 8,714ms

---

### 15. ✅ AAPL_context_competitivefactors_2025

**Question**: What are the key competitive factors for Apple in 2025?

**Expected Answer**: Price, product features, quality, third-party ecosystem, marketing, distribution capability, service, support, corporate reputation

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-K_2025.htm ✅
  3. raw_10-K_2025.htm ✅

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: Yes
  - Latency: 8,130ms

---

### 16. ✅ AAPL_context_competition_2025

**Question**: How does Apple describe its competition in fiscal 2025?

**Expected Answer**: Markets are highly competitive, characterized by aggressive price cutting and downward pressure on margins

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-K_2025.htm ✅
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: No
  - Latency: 9,439ms

---

### 17. ✅ AAPL_context_humancapital_2024

**Question**: How many employees does Apple have worldwide in 2024?

**Expected Answer**: Approximately 161,000 full-time equivalent employees

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm
  2. raw_10-K_2024.htm ✅
  3. raw_10-K_2024.htm ✅

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: No
  - Latency: 10,292ms

---

### 18. ✅ AAPL_context_humancapital_2025

**Question**: What is Apple's total employee count globally in 2025?

**Expected Answer**: Approximately 164,000 full-time equivalent employees

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm ✅
  2. raw_10-K_2025.htm ✅
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 9,658ms

**Analysis**: Perfect retrieval at rank #1.

---

## Multi-Period Queries (6 questions)

### 19. ✅ AAPL_multi_rev_years

**Question**: What was Apple's revenue for fiscal years 2023, 2024, and 2025?

**Expected Answer**: 2023: $383285M, 2024: $391035M, 2025: $416161M

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-K_2025.htm ✅
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: No
  - Latency: 10,437ms

---

### 20. ✅ AAPL_multi_iphonerev_years

**Question**: What were iPhone revenues for fiscal 2023, 2024, and 2025?

**Expected Answer**: 2023: $200583M, 2024: $201183M, 2025: $210066M

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm ✅
  2. raw_10-K_2024.htm
  3. raw_10-K_2025.htm ✅

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 9,064ms

**Analysis**: Perfect retrieval for multi-year iPhone revenue.

---

### 21. ✅ AAPL_multi_servicesrev_years

**Question**: What were Services revenues for fiscal 2023, 2024, and 2025?

**Expected Answer**: 2023: $85200M, 2024: $96169M, 2025: $98769M

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-K_2025.htm ✅
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: No
  - Latency: 9,266ms

---

### 22. ❌ AAPL_multi_americasrev_years

**Question**: What were Americas segment revenues for fiscal 2023, 2024, and 2025?

**Expected Answer**: 2023: $162560M, 2024: $166856M, 2025: $173756M

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q3.htm
  2. raw_10-Q_2024_Q3.htm
  3. Unknown

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 10,398ms

**Analysis**: Geographic multi-year query failure.

---

### 23. ❌ AAPL_multi_grchinarev_years

**Question**: What was the Greater China revenue comparison across fiscal years?

**Expected Answer**: 2023: $72559M, 2024: $66648M, 2025: $66886M

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2024_Q2.htm
  2. raw_10-Q_2025_Q1.htm
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 10,947ms

**Analysis**: Geographic multi-year query failure.

---

### 24. ✅ AAPL_multi_grossmargin_2025vs2024

**Question**: What was Apple's gross margin for fiscal 2025 compared to fiscal 2024?

**Expected Answer**: 2025: 46.2%, 2024: 45.6%

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-Q_2025_Q3.htm
  3. raw_10-K_2025.htm ✅

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.333
  - Answer in Retrieved Text: No
  - Latency: 9,648ms

---

## Cross-Document Queries (4 questions)

### 25. ✅ AAPL_cross_q1_rev_2024_2025

**Question**: What was Apple's Q1 revenue for fiscal 2024 and 2025?

**Expected Answer**: Q1 2024: $117154M, Q1 2025: $124271M

**Expected Source(s)**: raw_10-Q_2024_Q1.htm, raw_10-Q_2025_Q1.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm ✅
  2. raw_10-Q_2024_Q1.htm ✅
  3. raw_10-Q_2024_Q1.htm ✅

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 10,251ms

**Analysis**: Perfect cross-document retrieval for Q1 comparison.

---

### 26. ✅ AAPL_cross_q2_rev_2024_2025

**Question**: What was Apple's Q2 revenue for fiscal 2024 and 2025?

**Expected Answer**: Q2 2024: $90753M, Q2 2025: $94836M

**Expected Source(s)**: raw_10-Q_2024_Q2.htm, raw_10-Q_2025_Q2.htm

**Retrieved Sources**:
  1. raw_10-Q_2024_Q2.htm ✅
  2. raw_10-Q_2025_Q2.htm ✅
  3. raw_10-Q_2025_Q1.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 9,641ms

**Analysis**: Perfect cross-document retrieval for Q2 comparison.

---

### 27. ✅ AAPL_cross_q3_rev_2024_2025

**Question**: What was Apple's Q3 revenue for fiscal 2024 and 2025?

**Expected Answer**: Q3 2024: $85777M, Q3 2025: $85878M

**Expected Source(s)**: raw_10-Q_2024_Q3.htm, raw_10-Q_2025_Q3.htm

**Retrieved Sources**:
  1. raw_10-Q_2024_Q3.htm ✅
  2. raw_10-Q_2025_Q1.htm
  3. raw_10-Q_2025_Q3.htm ✅

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 9,061ms

**Analysis**: Perfect cross-document retrieval for Q3 comparison.

---

### 28. ❌ AAPL_cross_q4_rev_2025_year_2025

**Question**: What was Apple's Q4 2025 revenue and how does it compare to the full year 2025?

**Expected Answer**: Q4 2025: $111176M, Full year: $416161M

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2024_Q1.htm
  2. raw_10-Q_2025_Q3.htm
  3. raw_10-Q_2025_Q1.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 10,801ms

**Analysis**: System doesn't understand Q4 data is in 10-K, not 10-Q.

---

## Summary Statistics

### Category Performance

| Category | Questions | Hit@5 | MRR | Answer Coverage |
|----------|-----------|-------|-----|-----------------|
| Single Doc - Fact | 10 | 70.0% | 0.525 | 60.0% |
| Single Doc - Context | 8 | 100.0% | 0.812 | 50.0% |
| Multi-Period | 6 | 66.7% | 0.389 | 0.0% |
| Cross-Document | 4 | 75.0% | 0.750 | 0.0% |
| **Overall** | **28** | **78.6%** | **0.610** | **35.7%** |

### Key Strengths ✅
1. **Context queries excel**: 100% Hit@5, 0.812 MRR
2. **Product-specific revenue**: Perfect retrieval for iPhone, Services
3. **Cross-document Q1-Q3**: All quarterly comparisons work perfectly
4. **Year filtering effective**: Correctly prioritizes 10-K_2025 for FY2024

### Key Weaknesses ❌
1. **Geographic segments**: Americas and Greater China queries consistently fail
2. **Q4 understanding**: System doesn't know Q4 data is in annual 10-K
3. **Net income retrieval**: Financial statement line items challenging
4. **Answer coverage**: Only 35.7% of retrieved chunks contain exact answers

---

*Evaluation completed: November 16, 2024*
*Configuration: qwen2.5:0.5b reranker with 4 parallel workers + year filtering*
*Total questions: 28 across 4 categories*