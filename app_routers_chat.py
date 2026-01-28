# app/routers/chat.py
"""
Chat endpoint for RAG system.

Handles user queries, executes RAG pipeline via LangGraph,
and returns responses with retrieved context.
"""

import time
import logging
from uuid import uuid4
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import asyncpg

from app.models import ChatRequest, ChatResponse, RetrievedDocument
from app.dependencies import get_db_pool, get_rag_graph, get_observability_manager
from graph.rag_graph import RAGGraph
from db.repositories.traces import TracesRepository
from observability.tracing import ObservabilityManager
from observability.metrics import MetricsPublisher

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    rag_graph: RAGGraph = Depends(get_rag_graph),
    obs_manager: ObservabilityManager = Depends(get_observability_manager),
) -> ChatResponse:
    """
    Execute RAG pipeline and return answer with retrieved context.
    
    Flow:
    1. Generate request ID
    2. Execute LangGraph RAG pipeline
    3. Format response
    4. Log trace to database
    5. Publish metrics
    
    Args:
        request: Chat request with query and optional metadata
        
    Returns:
        ChatResponse with answer, retrieved documents, and metadata
    """
    request_id = f"req_{uuid4().hex}"
    start_time = time.time()
    
    logger.info(
        "Processing chat request",
        extra={
            "request_id": request_id,
            "user_id": request.user_id,
            "query_length": len(request.query),
        }
    )
    
    try:
        # Trace the request through observability system
        async with obs_manager.trace_request(
            request_id=request_id,
            operation="chat",
            metadata={
                "user_id": request.user_id,
                "conversation_id": request.conversation_id,
                **request.metadata
            }
        ) as tracer:
            
            # Execute RAG graph
            graph_result = await rag_graph.ainvoke({
                "query": request.query,
                "messages": [],
                "metadata": {
                    "request_id": request_id,
                    "user_id": request.user_id,
                    **request.metadata
                }
            })
            
            # Extract results
            answer = graph_result.get("answer", "")
            reranked_docs = graph_result.get("reranked_documents", [])
            metadata = graph_result.get("metadata", {})
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Format retrieved documents
            retrieved_documents = [
                RetrievedDocument(
                    id=doc["id"],
                    text=doc["text"],
                    score=doc["score"],
                    metadata=doc["metadata"],
                    source_document_id=doc["source"]
                )
                for doc in reranked_docs
            ]
            
            # Create response
            response = ChatResponse(
                request_id=request_id,
                answer=answer,
                retrieved_documents=retrieved_documents,
                latency_ms=latency_ms,
                model_name=metadata.get("model", "unknown"),
                metadata={
                    **metadata,
                    "num_retrieved": len(retrieved_documents),
                }
            )
            
            # Log trace to database
            traces_repo = TracesRepository(db_pool)
            await traces_repo.create_trace(
                request_id=request_id,
                user_id=request.user_id,
                query=request.query,
                response=answer,
                retrieved_doc_ids=[str(doc.id) for doc in retrieved_documents],
                model_name=metadata.get("model"),
                model_version=metadata.get("model_version"),
                latency_ms=latency_ms,
                total_tokens=metadata.get("prompt_tokens", 0) + metadata.get("completion_tokens", 0),
                prompt_tokens=metadata.get("prompt_tokens"),
                completion_tokens=metadata.get("completion_tokens"),
                metadata={
                    **request.metadata,
                    "conversation_id": request.conversation_id,
                }
            )
            
            # Publish metrics
            # (metrics publisher would be injected as dependency)
            
            # Log result to tracer
            tracer.log_result(response.dict())
            
            logger.info(
                "Chat request completed",
                extra={
                    "request_id": request_id,
                    "latency_ms": latency_ms,
                    "num_retrieved": len(retrieved_documents),
                }
            )
            
            return response
            
    except Exception as e:
        logger.error(
            f"Error processing chat request: {e}",
            exc_info=True,
            extra={"request_id": request_id}
        )
        
        # Log error trace
        try:
            traces_repo = TracesRepository(db_pool)
            await traces_repo.create_trace(
                request_id=request_id,
                user_id=request.user_id,
                query=request.query,
                error=str(e),
                metadata=request.metadata
            )
        except Exception as log_error:
            logger.error(f"Failed to log error trace: {log_error}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    rag_graph: RAGGraph = Depends(get_rag_graph),
) -> StreamingResponse:
    """
    Streaming version of chat endpoint.
    
    Returns Server-Sent Events (SSE) stream of tokens.
    Useful for real-time UI updates.
    """
    request_id = f"req_{uuid4().hex}"
    
    async def generate_stream():
        """Generator for SSE stream."""
        try:
            # Execute graph with streaming
            async for chunk in rag_graph.astream({
                "query": request.query,
                "messages": [],
                "metadata": {"request_id": request_id}
            }):
                # Extract answer chunks if available
                if "answer" in chunk:
                    yield f"data: {chunk['answer']}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming: {e}", exc_info=True)
            yield f"data: {{'error': '{str(e)}'}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/history/{user_id}")
async def get_chat_history(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
) -> Dict[str, Any]:
    """
    Retrieve chat history for a user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of results
        offset: Pagination offset
        
    Returns:
        List of previous chat interactions
    """
    traces_repo = TracesRepository(db_pool)
    
    try:
        traces = await traces_repo.get_user_traces(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "user_id": user_id,
            "traces": [
                {
                    "request_id": trace.request_id,
                    "query": trace.query,
                    "response": trace.response,
                    "created_at": trace.created_at.isoformat(),
                    "latency_ms": trace.latency_ms,
                }
                for trace in traces
            ],
            "limit": limit,
            "offset": offset,
        }
        
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
