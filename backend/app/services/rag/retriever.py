from typing import List, Dict, Any, Optional
from app.services.rag.vector_store import vector_store
from app.services.rag.document_processor import document_processor
import logging

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retrieval-Augmented Generation retriever service"""

    def __init__(self):
        self.vector_store = vector_store
        self.document_processor = document_processor

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

            # Search vector store
            results = self.vector_store.search(
                query=query,
                n_results=n_results,
                where=where_filter
            )

            # Format results
            contexts = []
            if results["documents"] and len(results["documents"]) > 0:
                for i, doc in enumerate(results["documents"][0]):
                    contexts.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0,
                    })

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
