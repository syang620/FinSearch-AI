import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.services.rag.embeddings import embedding_service
import logging
import uuid

logger = logging.getLogger(__name__)


class VectorStore:
    """Service for managing document embeddings in ChromaDB"""

    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            logger.info("Initializing ChromaDB client")
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"description": "Financial documents and research papers"}
            )
            logger.info(f"Collection '{settings.CHROMA_COLLECTION_NAME}' initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store

        Args:
            texts: List of text chunks
            metadatas: List of metadata dicts for each chunk
            ids: Optional list of IDs (will be generated if not provided)

        Returns:
            List of document IDs
        """
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = embedding_service.embed_texts(texts)

            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(texts)} documents to vector store")
            return ids

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents

        Args:
            query: Search query
            n_results: Number of results to return
            where: Optional filter conditions

        Returns:
            Dictionary with search results
        """
        try:
            # Generate query embedding
            query_embedding = embedding_service.embed_text(query)

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise

    def delete_documents(self, ids: List[str]):
        """Delete documents by IDs"""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "document_count": count
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise


# Singleton instance
vector_store = VectorStore()
