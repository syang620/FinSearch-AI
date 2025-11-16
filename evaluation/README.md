# FinSearch AI Evaluation Dataset

## Overview
This directory contains evaluation datasets for testing the retrieval quality of the FinSearch AI RAG system. The primary dataset focuses on Apple Inc. financial data from SEC filings.

## Dataset: `retrieval_eval_dataset.json`

### Purpose
This dataset is designed to evaluate the retrieval accuracy of the RAG system by testing its ability to:
1. Find specific financial facts from SEC filings
2. Retrieve relevant contextual information
3. Handle multi-period comparisons
4. Process cross-document queries

### Dataset Structure

Each entry in the JSON array contains:
- **id**: Unique identifier for the question
- **question**: The natural language query to test
- **company**: Company name (currently Apple Inc.)
- **filings**: List of source documents containing the answer
  - form: Filing type (10-K, 10-Q)
  - year: Fiscal year
  - source: Document identifier
- **answer**: Ground truth answer
  - value: Numeric value, list, or structured data
  - unit: Unit of measurement (if applicable)
  - type: Answer type (numeric, text, list_text, structured_numeric, dict_text)
- **category**: Question category (see below)
- **is_unanswerable**: Whether the question should be answerable from the corpus

### Question Categories

1. **single_doc_fact** (10 questions)
   - Simple factual questions from a single document
   - Examples: Total net sales, net income, segment revenue

2. **single_doc_context** (8 questions)
   - Contextual information requiring text understanding
   - Examples: Product descriptions, competitive factors, employee count

3. **single_doc_multi_period** (6 questions)
   - Multi-year comparisons within a single document
   - Examples: 3-year revenue trends, gross margin comparisons

4. **cross_doc_fact** (4 questions)
   - Facts requiring information from multiple documents
   - Examples: Quarterly comparisons across different 10-Qs

### Evaluation Metrics

The dataset can be used to evaluate:

1. **Retrieval Precision**: Are the retrieved chunks relevant to the question?
2. **Retrieval Recall**: Does the system find all necessary information?
3. **Source Accuracy**: Are the correct documents identified?
4. **Answer Coverage**: Can the retrieved context answer the question?

### Usage Example

```python
import json
from app.services.rag.retriever import rag_retriever

# Load evaluation dataset
with open('retrieval_eval_dataset.json', 'r') as f:
    eval_data = json.load(f)

# Test retrieval for each question
for item in eval_data:
    question = item['question']
    company = item['company']

    # Retrieve context
    results = rag_retriever.retrieve_context(
        query=question,
        n_results=5,
        company_filter='AAPL'  # For Apple-specific queries
    )

    # Evaluate if retrieved context contains the answer
    # Compare with item['answer'] for accuracy
```

### Dataset Statistics
- Total Questions: 28
- Companies: Apple Inc.
- Filing Types: 10-K (2024, 2025), 10-Q (Q1-Q4 for 2024-2025)
- Answer Types:
  - Numeric: 17 questions
  - Text/Contextual: 5 questions
  - Structured/Multi-value: 6 questions

### Ground Truth Sources
All answers are derived from actual Apple SEC filings:
- 2024 Form 10-K
- 2025 Form 10-K
- Quarterly 10-Q reports for fiscal 2024-2025

### Future Expansions
- Add more companies (Microsoft, Tesla, etc.)
- Include more complex reasoning questions
- Add negative test cases (unanswerable questions)
- Include questions requiring mathematical calculations