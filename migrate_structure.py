#!/usr/bin/env python3
"""
FinSearch-AI Structure Migration Script
Automates the initial migration to data science-centric structure
"""

import os
import shutil
from pathlib import Path
import json
import argparse
from typing import List, Tuple

class StructureMigrator:
    def __init__(self, root_path: Path, dry_run: bool = True):
        self.root = root_path
        self.dry_run = dry_run
        self.actions_log = []

    def log_action(self, action: str, source: str, dest: str = None):
        """Log migration actions"""
        msg = f"{action}: {source}"
        if dest:
            msg += f" → {dest}"
        self.actions_log.append(msg)
        if self.dry_run:
            print(f"[DRY RUN] {msg}")
        else:
            print(f"[EXECUTE] {msg}")

    def create_new_structure(self):
        """Create the new directory structure"""
        directories = [
            # Data directories
            "data/raw/edgar",
            "data/raw/earnings_calls",
            "data/interim/normalized",
            "data/processed/embeddings",
            "data/processed/chunks",
            "data/processed/indexes",

            # Source code
            "src/finsearch/config",
            "src/finsearch/data",
            "src/finsearch/features",
            "src/finsearch/models",
            "src/finsearch/evaluation",
            "src/finsearch/utils",

            # Models and reports
            "models/artifacts/embeddings",
            "models/artifacts/rerankers",
            "models/checkpoints",
            "reports/figures",
            "reports/tables",

            # Configs and experiments
            "configs/experiments",
            "configs/hydra",
            "experiments/runs",
            "experiments/results",

            # Tests
            "tests/unit",
            "tests/integration",

            # Scripts and notebooks
            "scripts",
            "notebooks",
        ]

        for dir_path in directories:
            full_path = self.root / dir_path
            if not full_path.exists():
                if not self.dry_run:
                    full_path.mkdir(parents=True, exist_ok=True)
                self.log_action("CREATE_DIR", str(dir_path))

    def move_data_files(self):
        """Move data files to new locations"""
        moves = [
            # Raw data
            ("data/edgar", "data/raw/edgar"),
            ("data/earnings_calls", "data/raw/earnings_calls"),
            ("data/earnings_calls_manual", "data/raw/earnings_calls_manual"),
            ("data/sp500_companies.json", "data/raw/sp500_companies.json"),

            # Processed data
            ("data_parsed", "data/interim/normalized"),
            ("data_chunked", "data/processed/chunks"),
        ]

        for source, dest in moves:
            source_path = self.root / source
            dest_path = self.root / dest

            if source_path.exists():
                if not self.dry_run:
                    if source_path.is_file():
                        shutil.copy2(source_path, dest_path)
                    else:
                        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                self.log_action("MOVE", source, dest)

    def create_init_files(self):
        """Create __init__.py files for Python packages"""
        packages = [
            "src/finsearch",
            "src/finsearch/config",
            "src/finsearch/data",
            "src/finsearch/features",
            "src/finsearch/models",
            "src/finsearch/evaluation",
            "src/finsearch/utils",
        ]

        for package in packages:
            init_file = self.root / package / "__init__.py"
            if not init_file.exists():
                if not self.dry_run:
                    init_file.touch()
                self.log_action("CREATE_INIT", f"{package}/__init__.py")

    def create_pyproject_toml(self):
        """Create modern Python packaging file"""
        pyproject_content = '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "finsearch-ai"
version = "2.0.0"
description = "Financial document search and analysis with RAG"
readme = "README.md"
requires-python = ">=3.9"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]

dependencies = [
    # Core
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
    "hydra-core>=1.3.0",

    # ML/NLP
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "sentence-transformers>=2.2.0",
    "chromadb>=0.4.0",
    "langchain>=0.1.0",
    "openai>=1.0.0",
    "tiktoken>=0.5.0",

    # Data processing
    "beautifulsoup4>=4.12.0",
    "pypdf>=3.0.0",
    "rank-bm25>=0.2.2",

    # API (optional)
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",

    # Evaluation
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.4.0",
    "ipykernel>=6.25.0",
    "jupyter>=1.0.0",
    "nbformat>=5.9.0",
]

experiment = [
    "mlflow>=2.5.0",
    "wandb>=0.15.0",
    "optuna>=3.3.0",
]

[project.scripts]
finsearch-prepare = "scripts.prepare_data:main"
finsearch-evaluate = "scripts.evaluate_rag:main"
finsearch-serve = "scripts.serve_api:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100
target-version = ["py39"]

[tool.isort]
profile = "black"
line_length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=src/finsearch --cov-report=term-missing"
'''

        pyproject_path = self.root / "pyproject.toml"
        if not pyproject_path.exists():
            if not self.dry_run:
                pyproject_path.write_text(pyproject_content)
            self.log_action("CREATE", "pyproject.toml")

    def create_default_config(self):
        """Create default configuration file"""
        config_content = '''# Default configuration for FinSearch-AI

data:
  raw_path: data/raw
  interim_path: data/interim
  processed_path: data/processed
  chunk_size: 512
  chunk_overlap: 128

embeddings:
  model: "BAAI/bge-small-en-v1.5"
  dimension: 384
  batch_size: 32

retrieval:
  use_hybrid: true
  dense_weight: 0.7
  sparse_weight: 0.3
  top_k: 20

reranking:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
  top_k: 5

generation:
  model: "gpt-4"
  temperature: 0.1
  max_tokens: 1000

evaluation:
  metrics:
    - precision_at_k
    - recall_at_k
    - mrr
    - ndcg
  k_values: [1, 3, 5, 10, 20]

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
'''

        config_path = self.root / "configs" / "default.yaml"
        if not config_path.exists():
            if not self.dry_run:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(config_content)
            self.log_action("CREATE", "configs/default.yaml")

    def archive_old_structure(self):
        """Archive old structure components"""
        archive_dir = self.root / "_archive_old_structure"

        to_archive = [
            "frontend",
            "backend/app/api",  # Complex routing
            "cli_chat.py",
            "finsearch",
        ]

        for item in to_archive:
            source = self.root / item
            if source.exists():
                dest = archive_dir / item
                if not self.dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if source.is_file():
                        shutil.copy2(source, dest)
                    else:
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                self.log_action("ARCHIVE", item, f"_archive_old_structure/{item}")

    def generate_migration_report(self):
        """Generate a report of all migration actions"""
        report_path = self.root / "MIGRATION_REPORT.md"

        report_content = f"""# Migration Report

## Summary
- Total actions: {len(self.actions_log)}
- Mode: {'DRY RUN' if self.dry_run else 'EXECUTED'}
- Date: {import.datetime.datetime.now().isoformat()}

## Actions Performed

"""
        for action in self.actions_log:
            report_content += f"- {action}\n"

        report_content += """

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
"""

        if not self.dry_run:
            report_path.write_text(report_content)
        print(f"\n{'='*50}")
        print(report_content)

    def run_migration(self):
        """Execute the full migration"""
        print(f"Starting migration {'[DRY RUN]' if self.dry_run else '[EXECUTE]'}")
        print("=" * 50)

        # Step 1: Create new structure
        print("\n1. Creating new directory structure...")
        self.create_new_structure()

        # Step 2: Move data files
        print("\n2. Moving data files...")
        self.move_data_files()

        # Step 3: Create Python package files
        print("\n3. Creating Python package files...")
        self.create_init_files()

        # Step 4: Create modern config files
        print("\n4. Creating configuration files...")
        self.create_pyproject_toml()
        self.create_default_config()

        # Step 5: Archive old structure
        print("\n5. Archiving old structure...")
        self.archive_old_structure()

        # Step 6: Generate report
        print("\n6. Generating migration report...")
        self.generate_migration_report()

def main():
    parser = argparse.ArgumentParser(
        description="Migrate FinSearch-AI to data science-centric structure"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration (default is dry run)"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Root path of the project (default: current directory)"
    )

    args = parser.parse_args()

    migrator = StructureMigrator(
        root_path=args.path,
        dry_run=not args.execute
    )

    if not args.execute:
        print("=" * 50)
        print("DRY RUN MODE - No files will be modified")
        print("Add --execute flag to perform actual migration")
        print("=" * 50)
        response = input("\nContinue with dry run? (y/n): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            return
    else:
        print("=" * 50)
        print("EXECUTE MODE - Files will be modified!")
        print("=" * 50)
        response = input("\nAre you sure you want to proceed? (yes/no): ")
        if response.lower() != 'yes':
            print("Migration cancelled.")
            return

    migrator.run_migration()

    print("\n" + "=" * 50)
    if not args.execute:
        print("Dry run complete! Review the actions above.")
        print("Run with --execute flag to perform actual migration.")
    else:
        print("Migration complete! Check MIGRATION_REPORT.md for details.")

if __name__ == "__main__":
    import datetime
    main()