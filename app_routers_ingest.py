# app/routers/ingest.py
"""
Document ingestion endpoint.

Handles document upload, chunking, embedding, and indexing into Postgres.
"""

import logging
from typing import Dict, Any, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
import asyncpg

from app.models import IngestRequest, IngestResponse
from app.dependencies import get_db_pool, get_s3_client
from rag.ingestion import IngestionPipeline
from db.repositories.documents import DocumentsRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    s3_client = Depends(get_s3_client),
) -> IngestResponse:
    """
    Ingest a document from S3 into the RAG system.
    
    Process:
    1. Download document from S3
    2. Extract text
    3. Chunk with overlap
    4. Generate embeddings
    5. Store in Postgres with full-text search vector
    
    Args:
        request: Ingestion request with S3 key and metadata
        
    Returns:
        Ingestion status with number of chunks created
    """
    source_document_id = request.source_document_id or request.s3_key
    
    logger.info(
        "Starting document ingestion",
        extra={
            "source_document_id": source_document_id,
            "s3_key": request.s3_key,
            "force_reindex": request.force_reindex,
        }
    )
    
    try:
        # Check if document already exists
        docs_repo = DocumentsRepository(db_pool)
        existing_version = await docs_repo.get_latest_version(source_document_id)
        
        if existing_version is not None and not request.force_reindex:
            logger.info(
                f"Document already indexed at version {existing_version}",
                extra={"source_document_id": source_document_id}
            )
            return IngestResponse(
                source_document_id=source_document_id,
                chunks_created=0,
                version=existing_version,
                status="skipped_already_indexed"
            )
        
        # Initialize ingestion pipeline
        ingestion_pipeline = IngestionPipeline(
            db_pool=db_pool,
            s3_client=s3_client,
            bucket_name="rag-system-documents",
        )
        
        # Run ingestion (can be sync or async)
        result = await ingestion_pipeline.ingest_from_s3(
            s3_key=request.s3_key,
            source_document_id=source_document_id,
            metadata=request.metadata,
            force_reindex=request.force_reindex,
        )
        
        logger.info(
            "Document ingestion completed",
            extra={
                "source_document_id": source_document_id,
                "chunks_created": result["chunks_created"],
                "version": result["version"],
            }
        )
        
        return IngestResponse(
            source_document_id=source_document_id,
            chunks_created=result["chunks_created"],
            version=result["version"],
            status="success"
        )
        
    except Exception as e:
        logger.error(
            f"Error ingesting document: {e}",
            exc_info=True,
            extra={"source_document_id": source_document_id}
        )
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.post("/batch")
async def ingest_batch(
    s3_keys: List[str],
    background_tasks: BackgroundTasks,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    s3_client = Depends(get_s3_client),
) -> Dict[str, Any]:
    """
    Ingest multiple documents in batch.
    
    This endpoint queues documents for background processing
    to avoid timeout on large batches.
    
    Args:
        s3_keys: List of S3 keys to ingest
        
    Returns:
        Batch ingestion status
    """
    batch_id = f"batch_{uuid4().hex[:8]}"
    
    logger.info(
        f"Starting batch ingestion",
        extra={
            "batch_id": batch_id,
            "num_documents": len(s3_keys),
        }
    )
    
    # Queue each document for background processing
    for s3_key in s3_keys:
        background_tasks.add_task(
            ingest_document_background,
            s3_key=s3_key,
            db_pool=db_pool,
            s3_client=s3_client,
            batch_id=batch_id,
        )
    
    return {
        "batch_id": batch_id,
        "num_documents": len(s3_keys),
        "status": "queued",
        "message": "Documents queued for background ingestion"
    }


async def ingest_document_background(
    s3_key: str,
    db_pool: asyncpg.Pool,
    s3_client,
    batch_id: str,
):
    """Background task for document ingestion."""
    try:
        ingestion_pipeline = IngestionPipeline(
            db_pool=db_pool,
            s3_client=s3_client,
            bucket_name="rag-system-documents",
        )
        
        await ingestion_pipeline.ingest_from_s3(
            s3_key=s3_key,
            source_document_id=s3_key,
            metadata={"batch_id": batch_id},
        )
        
        logger.info(
            f"Background ingestion completed",
            extra={"s3_key": s3_key, "batch_id": batch_id}
        )
        
    except Exception as e:
        logger.error(
            f"Background ingestion failed: {e}",
            exc_info=True,
            extra={"s3_key": s3_key, "batch_id": batch_id}
        )


@router.delete("/document/{source_document_id}")
async def delete_document(
    source_document_id: str,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
) -> Dict[str, Any]:
    """
    Delete all chunks for a document.
    
    This soft-deletes by incrementing the version number,
    so old versions are preserved for audit/rollback.
    
    Args:
        source_document_id: Document identifier
        
    Returns:
        Deletion status
    """
    try:
        docs_repo = DocumentsRepository(db_pool)
        deleted_count = await docs_repo.soft_delete(source_document_id)
        
        logger.info(
            f"Document deleted",
            extra={
                "source_document_id": source_document_id,
                "chunks_deleted": deleted_count,
            }
        )
        
        return {
            "source_document_id": source_document_id,
            "chunks_deleted": deleted_count,
            "status": "deleted"
        }
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document/{source_document_id}")
async def get_document_info(
    source_document_id: str,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
) -> Dict[str, Any]:
    """
    Get information about an indexed document.
    
    Returns metadata, version, and chunk count.
    """
    try:
        docs_repo = DocumentsRepository(db_pool)
        
        chunks = await docs_repo.get_by_source_document(source_document_id)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "source_document_id": source_document_id,
            "version": chunks[0].version,
            "num_chunks": len(chunks),
            "metadata": chunks[0].metadata,
            "created_at": chunks[0].created_at.isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex-all")
async def reindex_all(
    background_tasks: BackgroundTasks,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    s3_client = Depends(get_s3_client),
) -> Dict[str, Any]:
    """
    Reindex all documents in S3.
    
    This is useful when:
    - Changing chunking strategy
    - Updating embedding model
    - Fixing indexing bugs
    
    WARNING: This is expensive and slow. Use with caution.
    """
    logger.warning("Full reindex triggered")
    
    # List all objects in S3 bucket
    paginator = s3_client.get_paginator('list_objects_v2')
    s3_keys = []
    
    async for page in paginator.paginate(Bucket="rag-system-documents"):
        for obj in page.get('Contents', []):
            s3_keys.append(obj['Key'])
    
    # Queue for batch ingestion
    batch_id = f"reindex_{uuid4().hex[:8]}"
    
    for s3_key in s3_keys:
        background_tasks.add_task(
            ingest_document_background,
            s3_key=s3_key,
            db_pool=db_pool,
            s3_client=s3_client,
            batch_id=batch_id,
        )
    
    logger.info(
        f"Reindex queued",
        extra={"batch_id": batch_id, "num_documents": len(s3_keys)}
    )
    
    return {
        "batch_id": batch_id,
        "num_documents": len(s3_keys),
        "status": "reindex_queued",
        "message": f"Reindexing {len(s3_keys)} documents"
    }
