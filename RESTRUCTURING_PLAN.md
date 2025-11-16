# FinSearch-AI Restructuring Plan: Data Science-Centric Architecture

## Objective
Transform the current bloated multi-service architecture into a lightweight, data science-centric structure focused on RAG pipeline experimentation and financial analysis.

## Core Principles
1. **Simplicity First**: Remove unnecessary abstractions and services
2. **Notebook-Driven Development**: Use Jupyter for exploration and prototyping
3. **Clear Data Flow**: raw → interim → processed pipeline
4. **Experiment Tracking**: Proper versioning of models and configurations
5. **API as Output**: Simple FastAPI wrapper only for serving, not core logic

## Proposed Structure

```
finsearch-ai/
├── README.md
├── pyproject.toml                     # Modern Python packaging
├── .gitignore
│
├── data/
│   ├── raw/                          # Original data (immutable)
│   │   ├── edgar/                    # SEC filings as downloaded
│   │   ├── earnings_calls/           # Raw transcripts
│   │   └── sp500_companies.json      # Company metadata
│   ├── interim/                      # Cleaned, normalized JSONL
│   │   └── normalized/               # Unified document format
│   └── processed/                    # ML-ready datasets
│       ├── embeddings/               # Pre-computed embeddings
│       ├── chunks/                   # Document chunks
│       └── indexes/                  # BM25, other indexes
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Explore raw financial data
│   ├── 02_chunking_analysis.ipynb    # Test chunking strategies
│   ├── 03_embedding_experiments.ipynb # Compare embedding models
│   ├── 04_retrieval_evaluation.ipynb  # RAG evaluation
│   ├── 05_reranker_comparison.ipynb   # Reranker ablation studies
│   └── 99_scratch.ipynb               # Experimental playground
│
├── src/
│   └── finsearch/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py           # Pydantic settings
│       │   └── schemas.yaml          # Data schemas
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py             # Load raw data
│       │   ├── parser.py             # Parse EDGAR/earnings
│       │   ├── normalizer.py         # Convert to unified format
│       │   └── validator.py          # Data quality checks
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── chunker.py            # Document chunking (simplified)
│       │   ├── embedder.py           # Embedding generation
│       │   └── metadata.py           # Metadata extraction
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── retriever.py          # Unified retrieval (dense+sparse)
│       │   ├── reranker.py           # Single optimized reranker
│       │   └── generator.py          # LLM response generation
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py            # Retrieval & generation metrics
│       │   ├── benchmark.py          # Evaluation pipelines
│       │   └── visualize.py          # Results plotting
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           └── io.py                 # File I/O utilities
│
├── models/
│   ├── artifacts/                    # Trained models
│   │   ├── embeddings/               # Embedding model files
│   │   └── rerankers/                # Fine-tuned rerankers
│   └── checkpoints/                  # Training checkpoints
│
├── reports/
│   ├── figures/                      # Performance plots
│   ├── tables/                       # Metrics tables
│   └── analysis.md                   # Key findings
│
├── scripts/
│   ├── prepare_data.py               # Data pipeline orchestrator
│   ├── train_reranker.py             # Reranker training
│   ├── evaluate_rag.py               # RAG evaluation
│   └── serve_api.py                  # Minimal FastAPI server
│
├── configs/
│   ├── default.yaml                  # Base configuration
│   ├── experiments/
│   │   ├── baseline.yaml             # Baseline RAG
│   │   ├── hybrid_search.yaml        # Hybrid retrieval
│   │   └── llm_reranker.yaml         # LLM-based reranking
│   └── hydra/                        # Hydra config management
│
├── experiments/
│   ├── runs/                         # MLflow/W&B experiment runs
│   └── results/                      # Evaluation results cache
│
└── tests/
    ├── unit/
    │   ├── test_chunker.py
    │   ├── test_retriever.py
    │   └── test_normalizer.py
    └── integration/
        └── test_rag_pipeline.py
```

## Migration Strategy

### Phase 1: Core Consolidation (Week 1)
1. **Merge backend services** into simplified `src/finsearch` modules
2. **Consolidate data processing scripts** into `scripts/prepare_data.py`
3. **Simplify RAG pipeline** - one retriever, one reranker
4. **Remove frontend** - move to separate repo or archive

### Phase 2: Data Pipeline (Week 2)
1. **Reorganize data directories** following raw/interim/processed pattern
2. **Create unified data loader** in `src/finsearch/data/`
3. **Standardize chunking** - pick best strategy from evaluation
4. **Pre-compute embeddings** for faster experimentation

### Phase 3: Experimentation Setup (Week 3)
1. **Create evaluation notebooks** for systematic comparisons
2. **Setup experiment tracking** with Hydra + MLflow
3. **Build benchmark dataset** from existing evaluation data
4. **Document baseline performance**

### Phase 4: API Simplification (Week 4)
1. **Create minimal FastAPI wrapper** in `scripts/serve_api.py`
2. **Remove complex routing** - single chat endpoint
3. **Optional dashboard** as separate Streamlit app
4. **Docker compose** for simple deployment

## What Gets Removed/Archived

### Remove Completely:
- Frontend folder (move to `finsearch-ui` repo if needed)
- Multiple reranker variants (keep only best performing)
- Complex service abstractions
- Redundant test files
- CLI chat interface (replaced by notebooks)

### Archive for Reference:
- Current evaluation results
- Existing chunking strategies (keep code, pick one for production)
- Multiple embedding experiments

### Consolidate:
- 23 root-level scripts → 4 focused scripts
- 4 reranker services → 1 optimized reranker
- Multiple chunking strategies → 1 configurable chunker
- Separate edgar/earnings ingestion → unified data loader

## Key Improvements

1. **50% Less Code**: Remove abstractions and redundant implementations
2. **Faster Iteration**: Notebook-based development with cached data
3. **Better Experiments**: Proper tracking and comparison framework
4. **Cleaner Dependencies**: Single `pyproject.toml` with optional groups
5. **Easier Onboarding**: Clear data flow and simple structure

## Configuration Management

Using Hydra for experiment management:
```yaml
# configs/default.yaml
data:
  chunk_size: 512
  overlap: 128

retrieval:
  top_k: 20
  use_hybrid: true

reranking:
  model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
  top_k: 5

generation:
  model: "gpt-4"
  temperature: 0.1
```

## Success Metrics
- [ ] Repository size reduced by 40%+
- [ ] Test coverage maintained at 80%+
- [ ] RAG performance unchanged or improved
- [ ] Development iteration speed 2x faster
- [ ] Clear separation of experimentation vs production code

## Next Steps
1. Create feature branch: `refactor/data-science-structure`
2. Start with Phase 1 consolidation
3. Maintain backward compatibility during transition
4. Update documentation progressively
5. Final cutover after validation

## Timeline
- **Week 1-2**: Core restructuring
- **Week 3**: Data pipeline and experiments
- **Week 4**: Testing and documentation
- **Week 5**: Deployment and cutover

This plan transforms FinSearch-AI from a complex multi-service application into a focused data science project optimized for RAG experimentation and financial analysis.