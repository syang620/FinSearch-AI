from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm.llm_service import llm_service
from app.services.rag.retriever import rag_retriever
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with optional RAG support

    Args:
        request: ChatRequest with query and options

    Returns:
        ChatResponse with AI response and sources
    """
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Get response with or without RAG
        if request.use_rag:
            # Retrieve context using RAG
            retrieval_results = rag_retriever.retrieve_context(
                query=request.query,
                n_results=5,
                company_filter=request.company_filter
            )

            # Build context for LLM
            if retrieval_results["contexts"]:
                context = "\n\n".join([
                    f"Source ({ctx['metadata'].get('filename', 'Unknown')}):\n{ctx['text']}"
                    for ctx in retrieval_results["contexts"]
                ])

                # Generate response with context
                response = llm_service.chat(
                    query=request.query,
                    context=context,
                    max_new_tokens=512
                )

                sources = [
                    {
                        "text": ctx["text"][:200] + "...",  # Truncate for response
                        "metadata": ctx["metadata"],
                        "relevance_score": 1 - ctx["distance"]  # Convert distance to similarity
                    }
                    for ctx in retrieval_results["contexts"]
                ]
            else:
                # No context found, generate without RAG
                response = llm_service.chat(query=request.query, max_new_tokens=512)
                sources = []
        else:
            # Direct LLM response without RAG
            response = llm_service.chat(query=request.query, max_new_tokens=512)
            sources = None

        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            sources=sources,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
async def get_model_info():
    """Get information about the current LLM model"""
    try:
        return llm_service.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_rag_stats():
    """Get RAG system statistics"""
    try:
        return rag_retriever.get_stats()
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
