# FinSearch-AI 🔍📊

A lightweight, data science-centric RAG (Retrieval-Augmented Generation) system for financial document analysis. Optimized for experimentation and research with SEC filings and earnings call transcripts.

## 🎯 Overview

FinSearch-AI provides a streamlined pipeline for searching and analyzing financial documents using state-of-the-art retrieval and ranking techniques. The system has been restructured to focus on rapid experimentation and research iteration.

### Key Features

- **Hybrid Search**: Combines dense (vector) and sparse (BM25) retrieval
- **Advanced Reranking**: Cross-encoder models for improved relevance
- **Multiple Chunking Strategies**: Semantic, window, and section-based chunking
- **Unified Data Pipeline**: Clear flow from raw → interim → processed data
- **Notebook-First Development**: Jupyter notebooks for experimentation
- **Minimal API**: Simple FastAPI server for production serving

## 📁 Project Structure

```
finsearch-ai/
├── data/                      # Data pipeline
│   ├── raw/                   # Original documents
│   │   ├── edgar/             # SEC filings
│   │   └── earnings_calls/    # Transcripts
│   ├── interim/               # Normalized JSONL
│   └── processed/             # ML-ready data
│       ├── embeddings/        # Vector store
│       ├── chunks/            # Document chunks
│       └── indexes/           # BM25 indices
│
├── src/finsearch/             # Core library
│   ├── config/                # Configuration
│   ├── data/                  # Data loading
│   ├── features/              # Feature engineering
│   ├── models/                # RAG components
│   └── evaluation/            # Metrics
│
├── notebooks/                 # Experimentation
│   └── 01_rag_pipeline_experiments.ipynb
│
├── scripts/                   # CLI tools
│   └── serve_api.py          # API server
│
└── configs/                   # Configurations
    └── default.yaml          # Default settings
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FinSearch-AI.git
cd FinSearch-AI

# Install dependencies (using modern Python packaging)
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,experiment]"
```

### Basic Usage

#### 1. Load and Explore Data

```python
from finsearch.data.loader import DataLoader

# Initialize loader
loader = DataLoader()

# List available companies
companies = loader.list_available_companies()
print(f"Available companies: {companies}")

# Load documents for a company
docs = loader.load_company_documents("AAPL")
print(f"Loaded {len(docs)} documents for Apple")
```

#### 2. Run RAG Pipeline

```python
from finsearch.models.retriever import HybridRetriever
from finsearch.models.reranker import Reranker

# Initialize retriever
retriever = HybridRetriever(
    use_hybrid=True,
    dense_weight=0.7,
    sparse_weight=0.3
)

# Initialize reranker
reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

# Search
query = "What are Apple's AI initiatives?"
retrieved = retriever.retrieve(query, k=20)
reranked = reranker.rerank(query, retrieved, top_k=5)

# Display results
for doc in reranked:
    print(f"Score: {doc.rerank_score:.3f}")
    print(f"Text: {doc.text[:200]}...\n")
```

#### 3. Start API Server

```bash
# Start the minimal API server
python scripts/serve_api.py

# With custom settings
python scripts/serve_api.py --host 0.0.0.0 --port 8080
```

API will be available at `http://localhost:8000`

Example request:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Apple's revenue?", "company": "AAPL"}'
```

## 📊 Experimentation

### Jupyter Notebooks

The project includes comprehensive notebooks for experimentation:

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01_rag_pipeline_experiments.ipynb
```

The notebook covers:
- Data exploration and visualization
- Chunking strategy comparison
- Embedding analysis
- Retrieval experiments
- Reranking evaluation
- End-to-end RAG pipeline

### Running Evaluations

```python
from finsearch.evaluation.metrics import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator(
    retriever=retriever,
    reranker=reranker,
    metrics=["precision", "recall", "mrr", "ndcg"]
)

# Load evaluation dataset
results, aggregated = evaluator.evaluate_from_file(
    "data/processed/benchmark.json",
    k_values=[1, 3, 5, 10, 20]
)

# Print summary
evaluator.print_summary(aggregated)
```

## ⚙️ Configuration

Configuration is managed through YAML files and environment variables:

```yaml
# configs/default.yaml
data:
  chunk_size: 512
  chunk_overlap: 128

retrieval:
  use_hybrid: true
  dense_weight: 0.7
  sparse_weight: 0.3
  top_k: 20

reranking:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
  top_k: 5
```

Override with environment variables:
```bash
export FINSEARCH_RETRIEVAL__TOP_K=30
export FINSEARCH_RERANKING__ENABLED=false
```

## 📈 Performance

Current benchmark results on financial Q&A dataset:

| Metric | @5 | @10 | @20 |
|--------|-----|------|------|
| Precision | 0.82 | 0.75 | 0.68 |
| Recall | 0.71 | 0.84 | 0.92 |
| MRR | 0.76 | - | - |
| NDCG | 0.79 | 0.81 | 0.85 |

## 🔧 Advanced Usage

### Custom Chunking Strategy

```python
from finsearch.features.chunker import DocumentChunker

# Initialize with custom strategy
chunker = DocumentChunker(
    strategy='semantic',  # or 'window', 'sentence', 'section'
    chunk_size=512,
    chunk_overlap=128
)

# Chunk document
chunks = chunker.chunk(document_text)

# Validate chunks
stats = chunker.validate_chunks(chunks)
print(f"Average chunk size: {stats['avg_size']:.1f} chars")
```

### Experiment Tracking

```python
# configs/experiments/my_experiment.yaml
retrieval:
  dense_weight: 0.5
  sparse_weight: 0.5

# Run with specific config
from finsearch.config import settings

config = settings.load_config("configs/experiments/my_experiment.yaml")
# ... run experiment with config
```

## 📚 Data Sources

The system currently includes data from 11 major companies:
- **Technology**: AAPL, MSFT, GOOGL, NVDA, META
- **E-commerce**: AMZN
- **Automotive**: TSLA
- **Financial**: JPM, V
- **Healthcare**: UNH
- **Semiconductors**: AVGO

Document types:
- 10-K Annual Reports
- 10-Q Quarterly Reports
- Earnings Call Transcripts

## 📚 Documentation

Complete documentation is available in the [`docs/`](./docs/) directory:

- **[Guides](./docs/guides/)** - Setup and usage guides
- **[Technical](./docs/technical/)** - Data dictionary and technical specs
- **[Migration](./docs/migration/)** - Restructuring documentation
- **[Experiments](./docs/experiments/)** - Experiment results and analysis

See the [Documentation Index](./docs/README.md) for a complete list.

## 🤝 Contributing

We welcome contributions! The simplified structure makes it easy to:

1. Add new retrieval strategies in `src/finsearch/models/`
2. Implement new chunking methods in `src/finsearch/features/`
3. Create evaluation notebooks in `notebooks/`
4. Add new data sources in `src/finsearch/data/`

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- SEC EDGAR for financial filings
- Hugging Face for embedding and reranking models
- ChromaDB for vector storage
- The RAG research community

---

**Note**: This is a research-focused implementation optimized for experimentation. For production deployments, additional considerations for scalability, security, and monitoring should be implemented.