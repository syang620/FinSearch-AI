# Migration Report

## Summary
- Total actions: 42
- Mode: EXECUTED
- Date: 2025-11-16T13:19:10.709398

## Actions Performed

- CREATE_DIR: data/raw/edgar
- CREATE_DIR: data/raw/earnings_calls
- CREATE_DIR: data/interim/normalized
- CREATE_DIR: data/processed/embeddings
- CREATE_DIR: data/processed/chunks
- CREATE_DIR: data/processed/indexes
- CREATE_DIR: src/finsearch/config
- CREATE_DIR: src/finsearch/data
- CREATE_DIR: src/finsearch/features
- CREATE_DIR: src/finsearch/models
- CREATE_DIR: src/finsearch/evaluation
- CREATE_DIR: src/finsearch/utils
- CREATE_DIR: models/artifacts/embeddings
- CREATE_DIR: models/artifacts/rerankers
- CREATE_DIR: models/checkpoints
- CREATE_DIR: reports/figures
- CREATE_DIR: reports/tables
- CREATE_DIR: configs/experiments
- CREATE_DIR: configs/hydra
- CREATE_DIR: experiments/runs
- CREATE_DIR: experiments/results
- CREATE_DIR: tests/unit
- CREATE_DIR: tests/integration
- CREATE_DIR: scripts
- CREATE_DIR: notebooks
- MOVE: data/edgar → data/raw/edgar
- MOVE: data/earnings_calls → data/raw/earnings_calls
- MOVE: data/earnings_calls_manual → data/raw/earnings_calls_manual
- MOVE: data/sp500_companies.json → data/raw/sp500_companies.json
- MOVE: data_parsed → data/interim/normalized
- MOVE: data_chunked → data/processed/chunks
- CREATE_INIT: src/finsearch/__init__.py
- CREATE_INIT: src/finsearch/config/__init__.py
- CREATE_INIT: src/finsearch/data/__init__.py
- CREATE_INIT: src/finsearch/features/__init__.py
- CREATE_INIT: src/finsearch/models/__init__.py
- CREATE_INIT: src/finsearch/evaluation/__init__.py
- CREATE_INIT: src/finsearch/utils/__init__.py
- CREATE: pyproject.toml
- CREATE: configs/default.yaml
- ARCHIVE: frontend → _archive_old_structure/frontend
- ARCHIVE: backend/app/api → _archive_old_structure/backend/app/api


## Next Steps

1. Review the migration report
2. Run consolidation scripts for Python modules
3. Update imports in existing code
4. Create notebooks from test scripts
5. Test the new structure
6. Update documentation

## Validation Commands

```bash
# Test imports
python -c "from src.finsearch import *"

# Run tests
pytest tests/

# Check data integrity
ls -la data/raw/
ls -la data/processed/
```
