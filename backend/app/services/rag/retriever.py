from typing import List, Dict, Any, Optional
from app.services.rag.vector_store import vector_store
from app.services.rag.document_processor import document_processor
from app.services.rag.hybrid_search import hybrid_search
from app.services.rag.reranker_service import reranker
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieval-Augmented Generation retriever service"""

    def __init__(self):
        self.vector_store = vector_store
        self.document_processor = document_processor
        self.hybrid_search = hybrid_search

    def ingest_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system

        Args:
            file_path: Path to the document
            metadata: Optional metadata (company, document_type, etc.)

        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"Ingesting document: {file_path}")

            # Process and chunk the document
            chunks = self.document_processor.process_and_chunk_file(file_path, metadata)

            # Extract texts and metadatas
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]

            # Add to vector store
            chunk_ids = self.vector_store.add_documents(texts, metadatas)

            logger.info(f"Successfully ingested {len(chunks)} chunks from {file_path}")

            return {
                "success": True,
                "file_path": file_path,
                "chunks_created": len(chunks),
                "chunk_ids": chunk_ids,
            }

        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            return {
                "success": False,
                "file_path": file_path,
                "error": str(e),
            }

    def retrieve_context(
        self,
        query: str,
        n_results: int = 5,
        company_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query

        Args:
            query: User query
            n_results: Number of chunks to retrieve
            company_filter: Optional company name filter

        Returns:
            Dictionary with retrieved chunks and metadata
        """
        try:
            # Build filter
            where_filter = None
            if company_filter:
                where_filter = {"company": company_filter}

            # Determine how many candidates to retrieve for reranking
            if settings.RERANKER_ENABLED:
                # Get more candidates for reranking
                initial_n_results = min(settings.RERANKER_CANDIDATE_POOL, n_results * 4)
            else:
                initial_n_results = n_results

            # Use hybrid search (BM25 + Vector) for initial retrieval
            results = self.hybrid_search.search(
                query=query,
                n_results=initial_n_results,
                where=where_filter,
                bm25_weight=0.5,  # Equal weights for BM25 and vector search
                dense_weight=0.5
            )

            # Format initial results
            candidates = []
            if results["documents"] and len(results["documents"]) > 0:
                for i, doc in enumerate(results["documents"][0]):
                    # Handle both 'scores' (from hybrid) and 'distances' (from vector) fields
                    score_or_distance = 0
                    if "scores" in results and results["scores"]:
                        score_or_distance = results["scores"][0][i] if i < len(results["scores"][0]) else 0
                    elif "distances" in results and results["distances"]:
                        score_or_distance = results["distances"][0][i] if i < len(results["distances"][0]) else 0

                    candidates.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": score_or_distance,
                    })

            # Apply reranking if enabled
            if settings.RERANKER_ENABLED and candidates:
                logger.info(f"Reranking {len(candidates)} candidates...")
                contexts = reranker.rerank(
                    query=query,
                    candidates=candidates,
                    top_k=n_results
                )
                logger.info(f"Reranked to top {len(contexts)} results")
            else:
                # Use candidates as-is, limited to n_results
                contexts = candidates[:n_results]

            logger.info(f"Retrieved {len(contexts)} contexts for query")

            return {
                "query": query,
                "contexts": contexts,
                "count": len(contexts),
            }

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise

    def build_prompt_with_context(
        self,
        query: str,
        n_results: int = 5,
        company_filter: Optional[str] = None
    ) -> str:
        """
        Build a prompt with retrieved context for LLM

        Args:
            query: User query
            n_results: Number of chunks to retrieve
            company_filter: Optional company filter

        Returns:
            Formatted prompt string
        """
        # Retrieve context
        retrieval_results = self.retrieve_context(query, n_results, company_filter)
        contexts = retrieval_results["contexts"]

        # Build prompt
        if not contexts:
            return f"Question: {query}\n\nAnswer:"

        context_text = "\n\n".join([
            f"Source {i+1} ({ctx['metadata'].get('filename', 'Unknown')}):\n{ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = f"""You are a financial research assistant. Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
{context_text}

Question: {query}

Answer:"""

        return prompt

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        return self.vector_store.get_collection_stats()


# Singleton instance
rag_retriever = RAGRetriever()
