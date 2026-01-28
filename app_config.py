# app/config.py
"""
Application configuration.

Loads settings from environment variables with sensible defaults.
Uses pydantic for validation and type safety.
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # Application
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Database
    DB_HOST: str = Field(..., env="DB_HOST")
    DB_PORT: int = Field(default=5432, env="DB_PORT")
    DB_NAME: str = Field(..., env="DB_NAME")
    DB_USER: str = Field(..., env="DB_USER")
    DB_PASSWORD: str = Field(..., env="DB_PASSWORD")
    DB_POOL_MIN_SIZE: int = Field(default=5, env="DB_POOL_MIN_SIZE")
    DB_POOL_MAX_SIZE: int = Field(default=20, env="DB_POOL_MAX_SIZE")
    
    # AWS
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    
    # S3
    S3_BUCKET_DOCUMENTS: str = Field(default="rag-system-documents", env="S3_BUCKET_DOCUMENTS")
    S3_BUCKET_ARTIFACTS: str = Field(default="rag-system-artifacts", env="S3_BUCKET_ARTIFACTS")
    S3_BUCKET_EVAL_DATASETS: str = Field(default="rag-system-eval-datasets", env="S3_BUCKET_EVAL_DATASETS")
    
    # Bedrock
    BEDROCK_REGION: str = Field(default="us-east-1", env="BEDROCK_REGION")
    BEDROCK_LLM_MODEL_ID: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        env="BEDROCK_LLM_MODEL_ID"
    )
    BEDROCK_EMBEDDING_MODEL_ID: str = Field(
        default="amazon.titan-embed-text-v2:0",
        env="BEDROCK_EMBEDDING_MODEL_ID"
    )
    
    # RAG Configuration
    RAG_CHUNK_SIZE: int = Field(default=512, env="RAG_CHUNK_SIZE")
    RAG_CHUNK_OVERLAP: int = Field(default=50, env="RAG_CHUNK_OVERLAP")
    RAG_TOP_K: int = Field(default=20, env="RAG_TOP_K")
    RAG_RERANK_TOP_K: int = Field(default=5, env="RAG_RERANK_TOP_K")
    RAG_VECTOR_WEIGHT: float = Field(default=0.6, env="RAG_VECTOR_WEIGHT")
    RAG_BM25_WEIGHT: float = Field(default=0.4, env="RAG_BM25_WEIGHT")
    
    # LLM Configuration
    LLM_TEMPERATURE: float = Field(default=0.0, env="LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = Field(default=2048, env="LLM_MAX_TOKENS")
    
    # Observability
    LANGSMITH_API_KEY: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = Field(default="rag-production", env="LANGSMITH_PROJECT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # CloudWatch
    CLOUDWATCH_NAMESPACE: str = Field(default="RAGSystem", env="CLOUDWATCH_NAMESPACE")
    ENABLE_CLOUDWATCH_METRICS: bool = Field(default=True, env="ENABLE_CLOUDWATCH_METRICS")
    
    # API
    API_RATE_LIMIT: int = Field(default=100, env="API_RATE_LIMIT")  # requests per minute
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    
    # Evaluation
    EVAL_DATASET_PATH: str = Field(
        default="s3://rag-system-eval-datasets/eval_v1.jsonl",
        env="EVAL_DATASET_PATH"
    )
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")  # For DeepEval
    
    # Security
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    API_KEY: Optional[str] = Field(default=None, env="API_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Singleton instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection for settings."""
    return settings
