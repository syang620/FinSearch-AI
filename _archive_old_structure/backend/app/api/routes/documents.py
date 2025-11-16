from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.schemas import DocumentResponse, DocumentMetadata, StatusResponse
from app.services.rag.retriever import rag_retriever
from pathlib import Path
from typing import Optional
import aiofiles
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Document upload directory
UPLOAD_DIR = Path("./data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    company: Optional[str] = Form(None),
    document_type: Optional[str] = Form(None)
):
    """
    Upload and ingest a document into the RAG system

    Args:
        file: The document file
        company: Optional company name
        document_type: Optional document type (e.g., '10-K', 'earnings_call')

    Returns:
        DocumentResponse with ingestion status
    """
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        # Save file
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{doc_id}{file_extension}"

        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        logger.info(f"Saved file: {file_path}")

        # Prepare metadata
        metadata = {
            "document_id": doc_id,
            "filename": file.filename,
            "upload_date": datetime.now().isoformat(),
        }

        if company:
            metadata["company"] = company
        if document_type:
            metadata["document_type"] = document_type

        # Ingest document into RAG system
        result = rag_retriever.ingest_document(str(file_path), metadata)

        if result["success"]:
            return DocumentResponse(
                document_id=doc_id,
                filename=file.filename,
                status="success",
                chunks_created=result["chunks_created"],
                message="Document uploaded and processed successfully"
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_documents():
    """List all uploaded documents"""
    try:
        # Get all files in upload directory
        files = list(UPLOAD_DIR.glob("*"))

        documents = []
        for file_path in files:
            if file_path.is_file():
                documents.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "upload_date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })

        return {
            "documents": documents,
            "count": len(documents)
        }

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}", response_model=StatusResponse)
async def delete_document(document_id: str):
    """
    Delete a document

    Args:
        document_id: Document ID to delete

    Returns:
        StatusResponse
    """
    try:
        # Find and delete file
        files = list(UPLOAD_DIR.glob(f"{document_id}.*"))

        if not files:
            raise HTTPException(status_code=404, detail="Document not found")

        for file_path in files:
            file_path.unlink()

        # Note: In a production system, you would also remove from vector store
        # For now, we'll just delete the file

        return StatusResponse(
            status="success",
            message=f"Document {document_id} deleted successfully"
        )

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
