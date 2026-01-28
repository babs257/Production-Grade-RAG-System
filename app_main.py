# app/main.py
"""
FastAPI application entry point for Production RAG System.

This module defines the main FastAPI app with all routers, middleware,
and lifecycle management.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncpg

from app.config import settings
from app.routers import chat, ingest, feedback, eval
from app.middleware import RequestLoggingMiddleware, ErrorHandlingMiddleware
from db.connection import create_db_pool, close_db_pool
from observability.logging import setup_logging

# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI application.
    
    Handles startup and shutdown events:
    - Database connection pool creation/teardown
    - Resource initialization/cleanup
    """
    # Startup
    logger.info("Starting RAG API server", extra={"version": settings.VERSION})
    
    try:
        # Initialize database pool
        app.state.db_pool = await create_db_pool(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            min_size=settings.DB_POOL_MIN_SIZE,
            max_size=settings.DB_POOL_MAX_SIZE,
        )
        logger.info("Database pool initialized")
        
        # Initialize other resources
        app.state.metrics_publisher = None  # Initialize if needed
        app.state.obs_manager = None  # Initialize if needed
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down RAG API server")
        
        if hasattr(app.state, "db_pool"):
            await close_db_pool(app.state.db_pool)
            logger.info("Database pool closed")


# Create FastAPI app
app = FastAPI(
    title="RAG Production System API",
    description="Production-grade RAG system with evaluation, monitoring, and feedback loops",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

# Include routers
app.include_router(chat.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(eval.router, prefix="/api/v1")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for load balancer.
    
    Checks:
    - API is running
    - Database connection is healthy
    """
    try:
        # Check database
        async with app.state.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        return {
            "status": "healthy",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
            }
        )


@app.get("/metrics", status_code=status.HTTP_200_OK)
async def metrics() -> Dict[str, Any]:
    """
    Prometheus-compatible metrics endpoint (optional).
    
    Returns application metrics for monitoring.
    """
    # TODO: Implement Prometheus metrics collection
    return {
        "requests_total": 0,
        "requests_duration_seconds": 0.0,
        "errors_total": 0,
    }


@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "RAG Production System API",
        "version": settings.VERSION,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info",
    )
