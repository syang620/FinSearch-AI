# Embedding Pipeline Guide

This guide explains how to generate embeddings for your financial documents using our unified embedding system that supports multiple providers.

## Overview

The embedding system is now modularized with a clean architecture:
- **`modules/embedder.py`** - Core embedding module with support for multiple providers
- **`embed_cli.py`** - Single CLI script for all embedding operations

## Supported Providers

1. **Ollama** - Local embeddings using Ollama models
2. **FinE5** - Abaci's financial-specialized embeddings (API, model: `abacinlp-text-v1`)
3. **SentenceTransformers** - GPU-optimized open models
4. **NV-Embed-v2** - NVIDIA's state-of-the-art embeddings (GPU-optimized)

## Prerequisites

```bash
# Install dependencies
pip install requests tqdm chromadb

# For FinE5: Get API key from https://abacinlp.com
# For Ollama: Install and run ollama locally
# For SentenceTransformers: pip install sentence-transformers torch
```

## Usage Examples

### Option 1: Complete Pipeline (Recommended)

Run the entire pipeline from chunks to ChromaDB in one command:

```bash
# Using FinE5 API
export ABACI_API_KEY="sk-your-api-key-here"
python scripts/embed_cli.py pipeline \
    --provider fine5 \
    --archive chunks_for_colab.tar.gz \
    --chroma-path data/chroma_db_fine5 \
    --reset

# Using Ollama (local)
python scripts/embed_cli.py pipeline \
    --provider ollama \
    --model nomic-embed-text \
    --data-path data/raw_test \
    --chroma-path data/chroma_db_ollama \
    --reset

# Using SentenceTransformers (GPU)
python scripts/embed_cli.py pipeline \
    --provider sentence-transformers \
    --model BAAI/bge-large-en-v1.5 \
    --data-path data/raw_test \
    --chroma-path data/chroma_db_bge \
    --reset
```

### Option 2: Step-by-Step

#### Step 1: Generate Embeddings

```bash
# FinE5 API
python scripts/embed_cli.py generate \
    --provider fine5 \
    --archive chunks_for_colab.tar.gz \
    --output fine5_embeddings.jsonl \
    --batch-size 16

**What this does:**
- Extracts chunks from tar.gz archive
- Parses document metadata (company, doc type, year, section)
- Calls Abaci FinE5 API in batches
- Saves results as JSONL with embeddings + metadata
- Shows cost estimation based on token usage

**Expected output:**
```
============================================================
FinE5 Embedding Pipeline
============================================================
Extracting chunks_for_colab.tar.gz to extracted_chunks...
  Processing: 10-K_2025.txt
  Processing: 10-K_2024.txt

Collected 20 chunks

Processing 20 chunks in batches of 16...
Embedding batches: 100%|████████| 2/2 [00:03<00:00]

=== Processing Summary ===
Total chunks: 20
Successful embeddings: 20
Failed embeddings: 0
API calls made: 2
Estimated tokens: 5,000
Estimated cost: $0.0500

✓ Embeddings saved to: fin_e5_embeddings.jsonl
✓ File size: 1.23 MB
```

## Step 2: Ingest to ChromaDB

```bash
# Load embeddings into ChromaDB
python scripts/ingest_fine5_to_chromadb.py \
    --input fin_e5_embeddings.jsonl \
    --chroma-path data/chroma_db_fine5 \
    --collection-name financial_documents \
    --reset
```

**What this does:**
- Loads JSONL file with pre-computed embeddings
- Creates new ChromaDB collection optimized for cosine similarity
- Bulk inserts all embeddings with complete metadata
- Verifies ingestion and tests retrieval

**Expected output:**
```
============================================================
ChromaDB FinE5 Ingestion
============================================================
Loading embeddings from fin_e5_embeddings.jsonl...
✓ Loaded 20 embedding records

Validation: Valid: 20 records, 1024-dimensional embeddings

✓ Initialized ChromaDB at data/chroma_db_fine5
✓ Collection: financial_documents

Ingesting 20 records in batches of 100...
Ingesting batches: 100%|████████| 1/1 [00:00<00:00]

=== Ingestion Summary ===
Total records: 20
Successful inserts: 20
Failed inserts: 0
Duplicate IDs skipped: 0
Success rate: 100.0%

=== Verification ===
Total documents in collection: 20

Sample of 5 documents:
1. ID: AAPL:10-K:2025:BUSINESS:0
   Company: AAPL
   Document: 10-K
   Year: 2025
   Section: BUSINESS
   Text preview: Item 1. Business Company Background The Company designs, manufactures and markets smartphones...

✓ Ingestion complete!
✓ ChromaDB ready at: data/chroma_db_fine5
```

## Step 3: Use in Your RAG System

Update your RAG configuration to use the new ChromaDB:

```python
# In your retriever configuration
from scripts.modules.retriever import ChromaDBRetriever

retriever = ChromaDBRetriever(
    chroma_path="data/chroma_db_fine5",
    collection_name="financial_documents",
    k=5
)

# Query example
results = retriever.retrieve("What was Apple's revenue in 2024?")
```

## Cost Estimation

FinE5 API pricing (example rates):
- $0.01 per 1,000 tokens
- Average chunk: ~250 tokens
- 1,000 chunks = ~250,000 tokens = ~$2.50

For your test dataset (~20 chunks):
- Estimated cost: ~$0.05

## Advantages of FinE5

1. **Financial Domain Expertise**: Trained specifically on financial documents
2. **Better Accuracy**: Understands financial terminology and context
3. **Instruction Prompting**: Built-in support for task-specific retrieval optimization
4. **No GPU Required**: API-based, runs anywhere
5. **Cost Effective**: Only pay for what you use
6. **High Quality**: 4096-dimensional embeddings optimized for retrieval

### Instruction Prompting

FinE5 supports instruction prompting for improved retrieval quality:

```python
# Automatically enabled by default with financial document task
embedder = create_embedder("fine5", api_key="sk-...")

# Customize the task description
embedder = create_embedder(
    "fine5",
    api_key="sk-...",
    task_description="Given a financial question, retrieve user replies that best answer the question."
)

# Disable instruction prompting (use raw text)
embedder = create_embedder("fine5", api_key="sk-...", use_instruction=False)
```

## Troubleshooting

### API Key Issues
```bash
# Make sure key is set
echo $ABACI_API_KEY

# Or pass directly
python scripts/build_fine5_embeddings.py --api-key "sk-..."
```

### Rate Limiting
The script includes:
- Automatic retry with exponential backoff
- Rate limiting (1 second between batches)
- Adjustable batch size

### Memory Issues
If processing large datasets:
- Reduce `--batch-size` (e.g., 8 instead of 16)
- Process in chunks using offset/limit

## Next Steps

1. **Production Setup**: Store API key securely (e.g., AWS Secrets Manager)
2. **Batch Processing**: Set up scheduled jobs for new documents
3. **Monitoring**: Track API usage and costs
4. **Optimization**: Fine-tune batch sizes based on your rate limits