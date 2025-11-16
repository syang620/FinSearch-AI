# FinSearch AI - Evaluation Results Review

**Date**: 2025-11-15T13:23:21.780336

**Total Questions**: 28

**Successful Evaluations**: 28

---

## Overall Performance

- **Hit Rate@5**: 60.7%
- **MRR**: 0.361
- **Precision@5**: 21.4%
- **Answer Coverage**: 28.6%
- **Avg Latency**: 203ms

## Performance by Category

### Single Doc Fact
- Questions: 10
- Hit Rate@5: 40.0%
- MRR: 0.208
- Answer Coverage: 40.0%

### Single Doc Context
- Questions: 8
- Hit Rate@5: 75.0%
- MRR: 0.410
- Answer Coverage: 50.0%

### Single Doc Multi Period
- Questions: 6
- Hit Rate@5: 50.0%
- MRR: 0.208
- Answer Coverage: 0.0%

### Cross Doc Fact
- Questions: 4
- Hit Rate@5: 100.0%
- MRR: 0.875
- Answer Coverage: 0.0%


---


## Single Document Fact Questions


### 1. ❌ AAPL_fact_rev_2024

**Question**: What were Apple's total net sales in fiscal 2024?

**Expected Answer**: 391035 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2024_Q1.htm
  2. raw_10-K_2024.htm
  3. raw_10-Q_2025_Q2.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 1768ms

---

### 2. ❌ AAPL_fact_rev_2025

**Question**: What were Apple's total net sales in fiscal 2025?

**Expected Answer**: 416161 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-Q_2025_Q2.htm
  3. raw_10-Q_2025_Q2.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 82ms

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
  - Latency: 223ms

---

### 4. ✅ AAPL_fact_netincome_2025

**Question**: What was Apple's net income in fiscal 2025?

**Expected Answer**: 112010 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm
  2. raw_10-Q_2025_Q1.htm
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.250
  - Answer in Retrieved Text: Yes
  - Latency: 80ms

---

### 5. ❌ AAPL_fact_servicesrev_2025

**Question**: In fiscal 2025, what were Apple's Services net sales?

**Expected Answer**: 109158 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-Q_2025_Q2.htm
  3. raw_10-Q_2025_Q2.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 226ms

---

### 6. ❌ AAPL_fact_iphonerev_2025

**Question**: In fiscal 2025, what were Apple's iPhone net sales?

**Expected Answer**: 209586 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-Q_2025_Q2.htm
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 80ms

---

### 7. ✅ AAPL_fact_americasrev_2024

**Question**: In fiscal 2024, what were Americas net sales?

**Expected Answer**: 167045 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q3.htm
  2. raw_10-Q_2024_Q1.htm
  3. raw_10-K_2025.htm ✓ CORRECT

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.333
  - Answer in Retrieved Text: Yes
  - Latency: 217ms

---

### 8. ❌ AAPL_fact_greaterchinarev_2025

**Question**: In fiscal 2025, what were Greater China net sales?

**Expected Answer**: 64377 million USD

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q2.htm
  2. raw_10-Q_2025_Q3.htm
  3. raw_10-Q_2025_Q1.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 79ms

---

### 9. ✅ AAPL_fact_distribution_split_2024

**Question**: In fiscal 2024, how were Apple's net sales split between direct and indirect distribution channels?

**Expected Answer**: (structured data)
  - direct: 38
  - indirect: 62

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2024.htm ✓ CORRECT
  2. raw_10-Q_2024_Q3.htm
  3. raw_10-K_2024.htm ✓ CORRECT

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: Yes
  - Latency: 231ms

---

### 10. ✅ AAPL_fact_distribution_split_2025

**Question**: In fiscal 2025, how were Apple's net sales split between direct and indirect distribution channels?

**Expected Answer**: (structured data)
  - direct: 40
  - indirect: 60

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-K_2025.htm ✓ CORRECT
  3. raw_10-Q_2025_Q2.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: Yes
  - Latency: 79ms

---

## Single Document Context Questions


### 1. ✅ AAPL_context_products_2024

**Question**: In the 2024 Form 10-K, which main product lines does Apple list under "Products"?

**Expected Answer**:
  - iPhone
  - Mac
  - iPad
  - Wearables, Home and Accessories

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-K_2025.htm
  3. raw_10-K_2025.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.200
  - Answer in Retrieved Text: Yes
  - Latency: 280ms

---

### 2. ✅ AAPL_context_wearables_2024

**Question**: How does Apple describe the "Wearables, Home and Accessories" category in the 2024 Form 10-K?

**Expected Answer**: Includes smartwatches (Apple Watch Ultra 2, Series 10, SE), wireless headphones (AirPods, AirPods Pro, AirPods Max, Beats), Apple Vision Pro spatial computer, Apple TV streaming device, HomePod and HomePod mini smart speakers, and other Apple-branded / third-party accessories.

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-Q_2024_Q3.htm
  3. raw_10-K_2025.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.250
  - Answer in Retrieved Text: No
  - Latency: 210ms

---

### 3. ❌ AAPL_context_segments_2024

**Question**: What geographic reportable segments does Apple disclose in the 2024 Form 10-K, and what regions do they cover?

**Expected Answer**: (structured data)
  - Americas: North and South America
  - Europe: European countries plus India, the Middle East and Africa
  - Greater China: China mainland, Hong Kong, Taiwan
  - Japan: Japan
  - Rest of Asia Pacific: Australia and other Asian countries not in the above segments

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm
  2. raw_10-Q_2025_Q1.htm
  3. raw_10-Q_2024_Q1.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: Yes
  - Latency: 79ms

---

### 4. ❌ AAPL_context_markets_2024

**Question**: According to the 2024 Form 10-K, which customer markets does Apple primarily serve?

**Expected Answer**:
  - Consumer
  - Small and mid-sized business
  - Education
  - Enterprise
  - Government

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-Q_2025_Q1.htm
  3. AAPL_FY2024_Q4.pdf

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: Yes
  - Latency: 220ms

---

### 5. ✅ AAPL_context_competitivefactors_2025

**Question**: In the 2025 Form 10-K, what key competitive factors does Apple highlight as important to its success?

**Expected Answer**:
  - price
  - product and service features (including security)
  - relative price/performance
  - quality and reliability
  - design and technology innovation
  - third-party ecosystem strength
  - marketing and distribution capability
  - service and support
  - corporate reputation
  - ability to protect and enforce IP rights

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-K_2025.htm ✓ CORRECT
  3. raw_10-K_2025.htm ✓ CORRECT

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: Yes
  - Latency: 214ms

---

### 6. ✅ AAPL_context_competition_2025

**Question**: In the 2025 Form 10-K, how does Apple characterize competitive intensity in its markets?

**Expected Answer**: Markets are highly competitive with aggressive price competition, downward margin pressure, frequent new products and services, short product life cycles, rapidly evolving standards, rapid adoption of technology by competitors, and competitors that imitate Apple's products, sometimes providing offerings at little or no profit or even at a loss.

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm ✓ CORRECT
  2. raw_10-Q_2025_Q1.htm
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 83ms

---

### 7. ✅ AAPL_context_humancapital_2024

**Question**: How many full-time equivalent employees did Apple report as of September 28, 2024?

**Expected Answer**: 164000 employees

**Expected Source(s)**: raw_10-K_2024.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-K_2025.htm
  3. raw_10-K_2024.htm ✓ CORRECT

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.333
  - Answer in Retrieved Text: No
  - Latency: 81ms

---

### 8. ✅ AAPL_context_humancapital_2025

**Question**: How many full-time equivalent employees did Apple report as of September 27, 2025?

**Expected Answer**: 166000 employees

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-K_2025.htm ✓ CORRECT
  2. raw_10-K_2025.htm ✓ CORRECT
  3. raw_10-K_2024.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 78ms

---

## Single Document Multi-Period Questions


### 1. ✅ AAPL_multi_rev_years

**Question**: For fiscal 2025, 2024, and 2023, what were Apple's total net sales in each year?

**Expected Answer**: (structured data)
  - 2025: 416161
  - 2024: 391035
  - 2023: 383285

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-K_2025.htm ✓ CORRECT
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: No
  - Latency: 81ms

---

### 2. ❌ AAPL_multi_iphonerev_years

**Question**: For fiscal 2025, 2024, and 2023, what were Apple's iPhone net sales?

**Expected Answer**: (structured data)
  - 2025: 209586
  - 2024: 201183
  - 2023: 200583

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-Q_2025_Q3.htm
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 79ms

---

### 3. ❌ AAPL_multi_servicesrev_years

**Question**: For fiscal 2025, 2024, and 2023, what were Apple's Services net sales?

**Expected Answer**: (structured data)
  - 2025: 109158
  - 2024: 96169
  - 2023: 85200

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-Q_2025_Q2.htm
  3. raw_10-Q_2025_Q3.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 78ms

---

### 4. ✅ AAPL_multi_americasrev_years

**Question**: For fiscal 2025, 2024, and 2023, what were Americas net sales?

**Expected Answer**: (structured data)
  - 2025: 178353
  - 2024: 167045
  - 2023: 162560

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q3.htm
  2. raw_10-K_2025.htm ✓ CORRECT
  3. raw_10-Q_2024_Q3.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: No
  - Latency: 80ms

---

### 5. ❌ AAPL_multi_grchinarev_years

**Question**: For fiscal 2025, 2024, and 2023, what were Greater China net sales?

**Expected Answer**: (structured data)
  - 2025: 64377
  - 2024: 66952
  - 2023: 72559

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q3.htm
  2. raw_10-Q_2024_Q1.htm
  3. raw_10-Q_2025_Q1.htm

**Metrics**:
  - Hit@5: No
  - MRR: 0.000
  - Answer in Retrieved Text: No
  - Latency: 215ms

---

### 6. ✅ AAPL_multi_grossmargin_2025vs2024

**Question**: What were Apple's total gross margin (in dollars) and gross margin percentage in fiscal 2025, and how did they compare with fiscal 2024?

**Expected Answer**: (structured data)
  - gross_margin_2025: {'value': 195201, 'unit': 'million USD', 'percentage': 46.9}
  - gross_margin_2024: {'value': 180683, 'unit': 'million USD', 'percentage': 46.2}

**Expected Source(s)**: raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q3.htm
  2. raw_10-K_2024.htm
  3. raw_10-Q_2024_Q1.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.250
  - Answer in Retrieved Text: No
  - Latency: 225ms

---

## Cross Document Fact Questions


### 1. ✅ AAPL_cross_q1_rev_2024_2025

**Question**: What were Apple's total net sales in Q1 of fiscal 2024 (three months ended December 30, 2023) and Q1 of fiscal 2025 (three months ended December 28, 2024)?

**Expected Answer**: (structured data)
  - Q1_2024: 119575
  - Q1_2025: 124300

**Expected Source(s)**: raw_10-Q_2024_Q1.htm, raw_10-Q_2025_Q1.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm ✓ CORRECT
  2. raw_10-Q_2024_Q3.htm
  3. raw_10-Q_2025_Q1.htm ✓ CORRECT

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 231ms

---

### 2. ✅ AAPL_cross_q2_rev_2024_2025

**Question**: What were Apple's total net sales in Q2 of fiscal 2025 (three months ended March 29, 2025) and the year-ago Q2 (three months ended March 30, 2024)?

**Expected Answer**: (structured data)
  - Q2_2024: 90753
  - Q2_2025: 95359

**Expected Source(s)**: raw_10-Q_2024_Q2.htm, raw_10-Q_2025_Q2.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q2.htm ✓ CORRECT
  2. raw_10-Q_2025_Q2.htm ✓ CORRECT
  3. raw_10-Q_2025_Q2.htm ✓ CORRECT

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 80ms

---

### 3. ✅ AAPL_cross_q3_rev_2024_2025

**Question**: What were Apple's total net sales in Q3 of fiscal 2025 (three months ended June 28, 2025) and Q3 of fiscal 2024 (three months ended June 29, 2024)?

**Expected Answer**: (structured data)
  - Q3_2024: 85777
  - Q3_2025: 94036

**Expected Source(s)**: raw_10-Q_2024_Q3.htm, raw_10-Q_2025_Q3.htm

**Retrieved Sources**:
  1. raw_10-Q_2024_Q3.htm ✓ CORRECT
  2. raw_10-Q_2025_Q3.htm ✓ CORRECT
  3. raw_10-Q_2025_Q3.htm ✓ CORRECT

**Metrics**:
  - Hit@5: Yes
  - MRR: 1.000
  - Answer in Retrieved Text: No
  - Latency: 79ms

---

### 4. ✅ AAPL_cross_q4_rev_2025_year_2025

**Question**: What were Apple's total net sales for Q4 2025 and for the full fiscal year 2025, according to the FY25 Q4 consolidated statements?

**Expected Answer**: (structured data)
  - Q4_2025: 102466
  - FY2025: 416161

**Expected Source(s)**: raw_10-K_2025.htm, raw_10-K_2025.htm

**Retrieved Sources**:
  1. raw_10-Q_2025_Q1.htm
  2. raw_10-K_2025.htm ✓ CORRECT
  3. raw_10-Q_2025_Q2.htm

**Metrics**:
  - Hit@5: Yes
  - MRR: 0.500
  - Answer in Retrieved Text: No
  - Latency: 218ms

---