# FinSearch-AI Migration Mapping

## File Movement Guide: Current → New Structure

### Data Directories

| Current Location | New Location | Action |
|-----------------|--------------|--------|
| `data/edgar/` | `data/raw/edgar/` | Move as-is |
| `data/earnings_calls/` | `data/raw/earnings_calls/` | Move as-is |
| `data/earnings_calls_manual/` | `data/raw/earnings_calls/` | Merge with auto |
| `data/sp500_companies.json` | `data/raw/sp500_companies.json` | Move |
| `data_parsed/` | `data/interim/normalized/` | Move & rename |
| `data_chunked/` | `data/processed/chunks/` | Move |
| ChromaDB files | `data/processed/embeddings/` | Export & save |
| BM25 indexes | `data/processed/indexes/` | Move |

### Core Python Modules

| Current Backend Module | New Location | Notes |
|-----------------------|--------------|-------|
| `backend/app/core/config.py` | `src/finsearch/config/settings.py` | Simplify |
| `backend/app/models/schemas.py` | `src/finsearch/config/schemas.yaml` | Convert to YAML |
| **Data Ingestion** |  |  |
| `backend/.../edgar/*` | `src/finsearch/data/parser.py` | Consolidate parsers |
| `backend/.../earnings/*` | `src/finsearch/data/parser.py` | Single parser module |
| `backend/.../file_storage.py` | `src/finsearch/utils/io.py` | Generic I/O |
| `backend/.../metadata_schema.py` | `src/finsearch/data/normalizer.py` | Part of normalizer |
| **Chunking** |  |  |
| `backend/.../chunker/*.py` | `src/finsearch/features/chunker.py` | Single configurable class |
| **RAG Components** |  |  |
| `backend/.../embeddings.py` | `src/finsearch/features/embedder.py` | Simplified |
| `backend/.../vector_store.py` | `src/finsearch/models/retriever.py` | Part of retriever |
| `backend/.../hybrid_search.py` | `src/finsearch/models/retriever.py` | Unified retriever |
| `backend/.../bm25_index.py` | `src/finsearch/models/retriever.py` | Integrated |
| `backend/.../reranker_service*.py` | `src/finsearch/models/reranker.py` | Single best version |
| **LLM Services** |  |  |
| `backend/.../llm_service.py` | `src/finsearch/models/generator.py` | Renamed |
| `backend/.../ollama_service.py` | `src/finsearch/models/generator.py` | Integrated |
| **Other Services** |  |  |
| `backend/.../sentiment_service.py` | Archive or `src/finsearch/features/` | Optional |
| `backend/.../metrics_service.py` | Archive or `src/finsearch/features/` | Optional |

### Root-Level Scripts

| Current Script | New Location | Purpose |
|---------------|--------------|---------|
| `chunk_documents.py` | `scripts/prepare_data.py` | Part of pipeline |
| `ingest_chunks_to_vectordb.py` | `scripts/prepare_data.py` | Integrated |
| `ingest_bm25_index.py` | `scripts/prepare_data.py` | Integrated |
| `ingest_to_vectordb.py` | Remove | Redundant |
| `ingest_mvp_data.py` | Remove | One-time script |
| `ingest_manual_transcripts.py` | `scripts/prepare_data.py` | Integrated |
| `enrich_metadata.py` | `src/finsearch/features/metadata.py` | Module |
| `migrate_to_normalized_format.py` | Archive | One-time migration |
| `validate_vectordb.py` | `tests/integration/` | Test |
| `validate_hybrid_search.py` | `tests/integration/` | Test |
| `test_embedding_demo.py` | `notebooks/03_embedding_experiments.ipynb` | Notebook |
| `test_hybrid_search.py` | `notebooks/04_retrieval_evaluation.ipynb` | Notebook |
| `check_edgar_format.py` | `src/finsearch/data/validator.py` | Validation module |

### Evaluation & Testing

| Current Location | New Location | Notes |
|-----------------|--------------|-------|
| `evaluation/run_evaluation.py` | `scripts/evaluate_rag.py` | Simplified |
| `evaluation/run_bm25_evaluation.py` | `scripts/evaluate_rag.py` | Integrated |
| `evaluation/retrieval_eval_dataset.json` | `data/processed/benchmark.json` | Benchmark data |
| `evaluation/evaluation_results_*.json` | `experiments/results/` | Results storage |
| `backend/tests/` | `tests/unit/` | Reorganize |
| `tests/` (root) | `tests/integration/` | Integration tests |

### API Transformation

| Current | New | Notes |
|---------|-----|-------|
| `backend/app/api/routes/*.py` | `scripts/serve_api.py` | Single file, minimal |
| `backend/app/main.py` | `scripts/serve_api.py` | ~100 lines total |
| Multiple endpoints | Single `/chat` endpoint | Simplified |
| Complex routing | Direct function calls | Remove abstractions |

### Frontend & CLI

| Current | Action | Alternative |
|---------|--------|------------|
| `frontend/` | Archive or separate repo | `finsearch-ui` repo |
| `cli_chat.py` | Remove | Use notebooks |
| `finsearch` CLI | Remove | Use scripts directly |

### Configuration Files

| Current | New | Notes |
|---------|-----|-------|
| `backend/requirements.txt` | `pyproject.toml` | Modern packaging |
| `frontend/package.json` | Remove | No frontend |
| `docker-compose.yml` | Simplify | Single container |
| `.env.example` | `configs/default.yaml` | YAML config |
| Multiple Dockerfiles | Single `Dockerfile` | Simplified |

## Code Consolidation Examples

### Example 1: Chunker Consolidation

**Before** (6 files):
```
backend/app/services/data_ingestion/chunker/
├── pack.py
├── runner.py
├── sections.py
├── tokenize.py
├── validate.py
└── window.py
```

**After** (1 file):
```python
# src/finsearch/features/chunker.py
class DocumentChunker:
    def __init__(self, strategy='semantic', chunk_size=512, overlap=128):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        if self.strategy == 'semantic':
            return self._semantic_chunk(text)
        elif self.strategy == 'window':
            return self._window_chunk(text)
        # etc...
```

### Example 2: Retriever Unification

**Before** (4 services):
```
backend/app/services/rag/
├── embeddings.py
├── vector_store.py
├── hybrid_search.py
└── bm25_index.py
```

**After** (1 module):
```python
# src/finsearch/models/retriever.py
class HybridRetriever:
    def __init__(self, embedder, vector_store, bm25_index):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def retrieve(self, query: str, k: int = 20) -> List[Document]:
        # Unified retrieval logic
        dense_results = self._dense_search(query, k)
        sparse_results = self._sparse_search(query, k)
        return self._hybrid_fusion(dense_results, sparse_results)
```

## Migration Commands

### Step 1: Create new structure
```bash
# Create new directory structure
mkdir -p {data/{raw,interim,processed},notebooks,src/finsearch/{config,data,features,models,evaluation,utils}}
mkdir -p {models/{artifacts,checkpoints},reports/{figures,tables},scripts,configs/experiments}
mkdir -p {experiments/{runs,results},tests/{unit,integration}}
```

### Step 2: Move data files
```bash
# Move raw data
mv data/edgar data/raw/
mv data/earnings_calls* data/raw/
mv data/sp500_companies.json data/raw/

# Move processed data
mv data_parsed/* data/interim/normalized/
mv data_chunked/* data/processed/chunks/
```

### Step 3: Consolidate Python modules
```bash
# This will require manual refactoring
# Use the mapping table above as guide
```

### Step 4: Create notebooks from test scripts
```python
# Convert test scripts to notebooks
# Example: test_embedding_demo.py → 03_embedding_experiments.ipynb
```

### Step 5: Update imports
```bash
# Find and replace imports
# Example: "from backend.app.services.rag import embeddings"
#       → "from finsearch.features import embedder"
```

## Validation Checklist

- [ ] All data files moved to new locations
- [ ] Core functionality preserved in simplified modules
- [ ] Tests pass with new structure
- [ ] Notebooks can import from `src/finsearch`
- [ ] Single API endpoint works
- [ ] Documentation updated
- [ ] Git history preserved (use `git mv` when possible)

## Rollback Plan

1. Keep current structure in `main` branch
2. Work in feature branch `refactor/data-science-structure`
3. Tag current version: `git tag v1.0-pre-refactor`
4. If issues arise, can always revert to tagged version