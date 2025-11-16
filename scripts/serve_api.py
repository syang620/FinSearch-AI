#!/usr/bin/env python3
"""
Minimal FastAPI server for FinSearch-AI.
Single endpoint for RAG-based chat with financial documents.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging

# Import our modules
from finsearch.config import settings
from finsearch.data.loader import DataLoader
from finsearch.features.chunker import DocumentChunker
from finsearch.models.retriever import HybridRetriever
from finsearch.models.reranker import Reranker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = settings.load_config()

# Initialize FastAPI app
app = FastAPI(
    title="FinSearch AI",
    description="Minimal RAG API for financial document search",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (global for reuse)
data_loader = None
retriever = None
reranker = None


class ChatRequest(BaseModel):
    """Chat request model"""
    query: str
    company: Optional[str] = None
    top_k: Optional[int] = 5
    use_reranking: Optional[bool] = True


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    components: Dict[str, str]


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global data_loader, retriever, reranker

    logger.info("Initializing FinSearch AI components...")

    try:
        # Initialize data loader
        data_loader = DataLoader(base_path=config.data.raw_path.replace("/raw", ""))

        # Initialize retriever
        retriever = HybridRetriever(
            collection_name="finsearch",
            embedding_model=config.embeddings.model,
            use_hybrid=config.retrieval.use_hybrid,
            dense_weight=config.retrieval.dense_weight,
            sparse_weight=config.retrieval.sparse_weight,
            persist_directory=config.data.processed_path + "/embeddings"
        )

        # Load existing indices if available
        bm25_path = Path(config.data.processed_path) / "indexes" / "bm25_index.pkl"
        if bm25_path.exists():
            retriever.load_bm25_index(str(bm25_path))
            logger.info("Loaded existing BM25 index")

        # Initialize reranker
        if config.reranking.enabled:
            reranker = Reranker(
                model_name=config.reranking.model,
                batch_size=16
            )
            logger.info("Initialized reranker")

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        components={
            "data_loader": "ready" if data_loader else "not initialized",
            "retriever": "ready" if retriever else "not initialized",
            "reranker": "ready" if reranker else "not initialized"
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for RAG-based Q&A.

    Process:
    1. Retrieve relevant documents
    2. Optionally rerank them
    3. Generate response based on context
    """
    try:
        # Validate request
        if not request.query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # If company specified, add to filter
        filter_dict = None
        if request.company:
            filter_dict = {"company": request.company}

        # Retrieve relevant documents
        logger.info(f"Processing query: {request.query}")
        retrieved_docs = retriever.retrieve(
            query=request.query,
            k=config.retrieval.top_k,
            filter_dict=filter_dict
        )

        if not retrieved_docs:
            return ChatResponse(
                answer="I couldn't find any relevant information for your query.",
                sources=[],
                metadata={"retrieved_count": 0}
            )

        # Rerank if enabled
        if request.use_reranking and reranker:
            reranked_docs = reranker.rerank(
                query=request.query,
                documents=retrieved_docs,
                top_k=request.top_k or config.reranking.top_k
            )
            final_docs = reranked_docs
        else:
            final_docs = retrieved_docs[:request.top_k or 5]

        # Prepare context from documents
        context_parts = []
        sources = []

        for i, doc in enumerate(final_docs[:request.top_k or 5]):
            context_parts.append(f"[Source {i+1}]\n{doc.text}\n")
            sources.append({
                "text": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                "metadata": doc.metadata,
                "score": doc.score if hasattr(doc, 'score') else 0.0
            })

        context = "\n".join(context_parts)

        # Generate response (simplified - in production, integrate with LLM)
        # For now, return a formatted summary of retrieved content
        answer = generate_answer(request.query, context, config)

        return ChatResponse(
            answer=answer,
            sources=sources,
            metadata={
                "retrieved_count": len(retrieved_docs),
                "reranked_count": len(final_docs) if request.use_reranking else 0,
                "company_filter": request.company
            }
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/companies")
async def list_companies():
    """List available companies"""
    try:
        companies = data_loader.list_available_companies()
        return {"companies": companies, "count": len(companies)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    """Get data statistics"""
    try:
        stats = data_loader.get_stats()
        retriever_stats = retriever.get_stats() if retriever else {}
        reranker_stats = reranker.get_stats() if reranker else {}

        return {
            "data": stats,
            "retriever": retriever_stats,
            "reranker": reranker_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_answer(query: str, context: str, config: settings.Settings) -> str:
    """
    Generate answer based on query and context.

    In production, this would call an LLM (OpenAI, Anthropic, etc.).
    For now, returns a structured summary.
    """
    # This is a placeholder - integrate with actual LLM
    # Example integration:
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.chat.completions.create(
    #     model=config.generation.model,
    #     messages=[
    #         {"role": "system", "content": "You are a financial analyst..."},
    #         {"role": "user", "content": f"Question: {query}\n\nContext: {context}"}
    #     ],
    #     temperature=config.generation.temperature,
    #     max_tokens=config.generation.max_tokens
    # )
    # return response.choices[0].message.content

    # Placeholder response
    return f"""Based on the retrieved financial documents, here's what I found:

Query: {query}

The search returned relevant information from financial filings and earnings calls.
The retrieved context contains detailed information that addresses your query.

Key points from the sources:
- The documents contain relevant financial data and disclosures
- Multiple sources were consulted to provide comprehensive information
- The information is based on official SEC filings and earnings transcripts

[Note: This is a placeholder response. In production, an LLM would generate a specific answer based on the retrieved context.]"""


def main():
    """Main entry point"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run FinSearch AI API server")
    parser.add_argument("--host", default=config.api_host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.api_port, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=config.api_workers, help="Number of workers")
    parser.add_argument("--reload", action="store_true", default=config.api_reload, help="Enable auto-reload")

    args = parser.parse_args()

    # Run the server
    logger.info(f"Starting FinSearch AI API on {args.host}:{args.port}")
    uvicorn.run(
        "serve_api:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload
    )


if __name__ == "__main__":
    main()