# Production-Grade RAG System Architecture
## AWS-Native, LangChain/LangGraph, with Real Evaluations & Monitoring

**Author**: Tech Lead / ML Platform Team  
**Stack**: Python 3.11+, LangChain, LangGraph, FastAPI, AWS Bedrock, RDS Postgres + pgvector, S3, ECS Fargate  
**Focus**: Production-ready, observable, evaluated, feedback-driven

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Repository Structure](#2-repository-structure)
3. [Data Models & Schemas](#3-data-models--schemas)
4. [RAG Pipeline Implementation](#4-rag-pipeline-implementation)
5. [Evaluation Framework](#5-evaluation-framework)
6. [Observability & Monitoring](#6-observability--monitoring)
7. [Feedback Loops](#7-feedback-loops)
8. [AWS Infrastructure](#8-aws-infrastructure)
9. [CI/CD Pipeline](#9-cicd-pipeline)
10. [Scalability & Future Extensions](#10-scalability--future-extensions)

---

## 1. High-Level Architecture

### 1.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Applications                      │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Load Balancer                     │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│              ECS Fargate (FastAPI + LangGraph)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   /chat      │  │   /ingest    │  │  /feedback   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                   │
│         └──────────────────┴──────────────────┘                  │
│                            │                                      │
│                    ┌───────▼───────┐                             │
│                    │  LangGraph    │                             │
│                    │  RAG Pipeline │                             │
│                    └───────┬───────┘                             │
└────────────────────────────┼─────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
    │ AWS Bedrock   │ │ RDS Postgres │ │      S3      │
    │ (Claude 3.5)  │ │  + pgvector  │ │  Documents   │
    │  Embeddings   │ │  + tsvector  │ │  Eval Sets   │
    └───────────────┘ └──────────────┘ └──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  CloudWatch     │
                    │  Logs & Metrics │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   LangSmith     │
                    │  (or similar)   │
                    │  Tracing/Obs    │
                    └─────────────────┘
```

### 1.2 Data Flow

**Query Path (Production)**:
1. User sends query → ALB → FastAPI `/chat` endpoint
2. FastAPI validates request, generates request_id
3. LangGraph executes RAG pipeline:
   - **Retrieve Node**: Hybrid search (BM25 + vector) in Postgres
   - **Rerank Node** (optional): Score and filter retrieved chunks
   - **Generate Node**: Call Bedrock with context + query
4. Response returned to user
5. Trace logged to CloudWatch + LangSmith
6. Request stored in `traces` table

**Ingestion Path**:
1. Documents uploaded to S3 (versioned bucket)
2. Trigger: S3 event → Lambda or scheduled ECS task
3. Ingest service:
   - Download document from S3
   - Chunk with overlap (e.g., 512 tokens, 50 overlap)
   - Generate embeddings via Bedrock
   - Compute TSVECTOR for full-text search
   - Write to `documents` table in Postgres
4. Index version updated in metadata

**Evaluation Path**:
1. Scheduled job or CI pipeline triggers `/eval/run`
2. Load evaluation dataset from S3 (JSONL)
3. For each eval example:
   - Run RAG pipeline
   - Compute DeepEval metrics (faithfulness, relevancy, recall)
4. Aggregate results, compare against thresholds
5. Store results in S3 + CloudWatch metrics
6. Fail CI if metrics regress

**Feedback Loop**:
1. User provides feedback via `/feedback` endpoint
2. Feedback stored in `feedback` table, linked to `trace_id`
3. Nightly batch job:
   - Query low-score or downvoted interactions
   - Export as new evaluation examples
   - Append to evaluation dataset in S3
4. Next evaluation run includes these examples

---

## 2. Repository Structure

```
rag-production-system/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Configuration (env vars, secrets)
│   ├── dependencies.py            # FastAPI dependencies (DB, clients)
│   ├── models.py                  # Pydantic request/response models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat.py                # /chat endpoint
│   │   ├── ingest.py              # /ingest endpoint
│   │   ├── feedback.py            # /feedback endpoint
│   │   └── eval.py                # /eval/* endpoints
│   └── middleware.py              # Logging, tracing, error handling
│
├── graph/
│   ├── __init__.py
│   ├── rag_graph.py               # LangGraph graph definition
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── retrieve.py            # Retrieval node (hybrid search)
│   │   ├── rerank.py              # Optional reranking node
│   │   └── generate.py            # Generation node (Bedrock LLM)
│   └── state.py                   # Graph state definition
│
├── rag/
│   ├── __init__.py
│   ├── chunking.py                # Text chunking strategies
│   ├── embeddings.py              # Bedrock embeddings wrapper
│   ├── retrieval.py               # Hybrid retrieval implementation
│   ├── ingestion.py               # Document ingestion pipeline
│   └── models.py                  # Internal data models (Document, Chunk)
│
├── eval/
│   ├── __init__.py
│   ├── datasets.py                # Evaluation dataset loader
│   ├── metrics.py                 # DeepEval + RAGAS metrics
│   ├── runner.py                  # Evaluation runner script
│   ├── thresholds.py              # Quality gate thresholds
│   └── datasets/
│       └── eval_v1.jsonl          # Evaluation examples
│
├── observability/
│   ├── __init__.py
│   ├── tracing.py                 # LangSmith / custom tracing
│   ├── logging.py                 # Structured logging setup
│   └── metrics.py                 # CloudWatch metrics publisher
│
├── feedback/
│   ├── __init__.py
│   ├── collector.py               # Feedback collection logic
│   └── processor.py               # Convert feedback → eval examples
│
├── db/
│   ├── __init__.py
│   ├── connection.py              # Postgres connection pool
│   ├── schema.sql                 # Database schema DDL
│   ├── migrations/                # Alembic migrations
│   │   └── ...
│   └── repositories/
│       ├── __init__.py
│       ├── documents.py           # Document CRUD
│       ├── traces.py              # Trace CRUD
│       └── feedback.py            # Feedback CRUD
│
├── infra/
│   ├── terraform/                 # or CDK/CloudFormation
│   │   ├── main.tf
│   │   ├── ecs.tf                 # ECS Fargate cluster
│   │   ├── rds.tf                 # RDS Postgres
│   │   ├── s3.tf                  # S3 buckets
│   │   ├── iam.tf                 # IAM roles and policies
│   │   └── secrets.tf             # Secrets Manager
│   └── docker/
│       └── Dockerfile             # Application container
│
├── tests/
│   ├── unit/
│   │   ├── test_chunking.py
│   │   ├── test_retrieval.py
│   │   └── test_graph.py
│   ├── integration/
│   │   ├── test_api.py
│   │   └── test_ingestion.py
│   └── eval/
│       └── test_evaluation.py
│
├── scripts/
│   ├── ingest_documents.py        # CLI for manual ingestion
│   ├── run_evaluation.py          # CLI for offline eval
│   └── export_feedback.py         # Export feedback to eval set
│
├── .github/
│   └── workflows/
│       ├── ci.yml                 # CI pipeline
│       └── deploy.yml             # Deployment pipeline
│
├── pyproject.toml                 # Poetry/pip dependencies
├── Dockerfile
├── docker-compose.yml             # Local development
├── .env.example
└── README.md
```

---

## 3. Data Models & Schemas

### 3.1 Postgres Schema

```sql
-- db/schema.sql

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Documents table (chunks with embeddings and full-text search)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_document_id TEXT NOT NULL,           -- S3 key or logical doc ID
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',       -- {"page": 1, "section": "intro", ...}
    embedding VECTOR(1024),                      -- Bedrock Titan embeddings (1024-dim)
    tsv TSVECTOR,                                -- Full-text search vector
    version INTEGER NOT NULL DEFAULT 1,          -- Index version for safe re-indexing
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_chunk UNIQUE (source_document_id, chunk_index, version)
);

-- Indexes for hybrid search
CREATE INDEX idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
CREATE INDEX idx_documents_tsv ON documents USING gin(tsv);
CREATE INDEX idx_documents_source ON documents(source_document_id);
CREATE INDEX idx_documents_version ON documents(version);
CREATE INDEX idx_documents_metadata ON documents USING gin(metadata);

-- Trigger to auto-update tsvector
CREATE TRIGGER tsvector_update_trigger
BEFORE INSERT OR UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION
tsvector_update_trigger(tsv, 'pg_catalog.english', text);

-- Traces table (request/response logs for observability)
CREATE TABLE traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id TEXT UNIQUE NOT NULL,
    user_id TEXT,
    query TEXT NOT NULL,
    response TEXT,
    retrieved_doc_ids UUID[],                    -- Array of document IDs retrieved
    model_name TEXT,
    model_version TEXT,
    latency_ms INTEGER,
    total_tokens INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd NUMERIC(10, 6),
    error TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_traces_request_id ON traces(request_id);
CREATE INDEX idx_traces_user_id ON traces(user_id);
CREATE INDEX idx_traces_created_at ON traces(created_at DESC);

-- Feedback table (user feedback on responses)
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID REFERENCES traces(id) ON DELETE CASCADE,
    user_id TEXT,
    rating INTEGER CHECK (rating IN (-1, 1)),   -- thumbs down / thumbs up
    comment TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_feedback_trace_id ON feedback(trace_id);
CREATE INDEX idx_feedback_rating ON feedback(rating);
CREATE INDEX idx_feedback_created_at ON feedback(created_at DESC);

-- Evaluation results table (track eval metrics over time)
CREATE TABLE evaluation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_run_id TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    model_version TEXT NOT NULL,
    metrics JSONB NOT NULL,                      -- {"faithfulness": 0.85, "relevancy": 0.90, ...}
    num_examples INTEGER,
    passed BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_eval_results_run_id ON evaluation_results(eval_run_id);
CREATE INDEX idx_eval_results_created_at ON evaluation_results(created_at DESC);
```

### 3.2 Pydantic Models

```python
# app/models.py
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    id: UUID
    text: str
    score: float
    metadata: Dict[str, Any]
    source_document_id: str


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    retrieved_documents: List[RetrievedDocument]
    latency_ms: int
    model_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    s3_key: str
    source_document_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    force_reindex: bool = False


class IngestResponse(BaseModel):
    source_document_id: str
    chunks_created: int
    version: int
    status: str


class FeedbackRequest(BaseModel):
    request_id: str
    rating: int = Field(..., ge=-1, le=1)  # -1 (down), 1 (up)
    comment: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    feedback_id: UUID
    status: str


class EvalRunRequest(BaseModel):
    dataset_path: Optional[str] = None  # S3 path to eval dataset
    model_version: Optional[str] = None
    num_samples: Optional[int] = None


class EvalRunResponse(BaseModel):
    eval_run_id: str
    metrics: Dict[str, float]
    num_examples: int
    passed: bool
    details: Dict[str, Any]
```

---

## 4. RAG Pipeline Implementation

### 4.1 LangGraph Graph Definition

```python
# graph/rag_graph.py
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from graph.nodes.retrieve import retrieve_node
from graph.nodes.rerank import rerank_node
from graph.nodes.generate import generate_node


class RAGState(TypedDict):
    """State passed through the RAG graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    retrieved_documents: List[dict]
    reranked_documents: List[dict]
    context: str
    answer: str
    metadata: dict


def build_rag_graph() -> StateGraph:
    """
    Build the RAG graph with retrieve → rerank → generate flow.
    
    Graph structure:
        START → retrieve → rerank → generate → END
    
    Optional conditional edges can be added for:
    - Query classification (routing to different retrievers)
    - Self-correction loops (if answer quality is poor)
    - Tool calling (for multi-step reasoning)
    """
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    
    # Define edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# Singleton graph instance
rag_graph = build_rag_graph()
```

### 4.2 Retrieve Node (Hybrid Search)

```python
# graph/nodes/retrieve.py
from typing import List, Dict
from rag.retrieval import HybridRetriever
from graph.state import RAGState


async def retrieve_node(state: RAGState) -> Dict:
    """
    Retrieve relevant documents using hybrid search.
    
    Combines:
    - Dense vector search (cosine similarity on embeddings)
    - Sparse BM25-style search (Postgres full-text search)
    
    Fusion strategy: Reciprocal Rank Fusion (RRF)
    """
    query = state["query"]
    
    # Initialize retriever (injected via dependency in production)
    retriever = HybridRetriever(
        vector_weight=0.6,
        bm25_weight=0.4,
        top_k=20,
        rerank_top_k=5
    )
    
    # Perform hybrid retrieval
    retrieved_docs = await retriever.retrieve(query)
    
    return {
        "retrieved_documents": [
            {
                "id": str(doc.id),
                "text": doc.text,
                "score": doc.score,
                "metadata": doc.metadata,
                "source": doc.source_document_id
            }
            for doc in retrieved_docs
        ],
        "metadata": {
            **state.get("metadata", {}),
            "num_retrieved": len(retrieved_docs)
        }
    }
```

```python
# rag/retrieval.py
import asyncpg
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from rag.embeddings import BedrockEmbeddings


@dataclass
class RetrievedDocument:
    id: str
    text: str
    score: float
    metadata: dict
    source_document_id: str


class HybridRetriever:
    """
    Hybrid retrieval combining vector similarity and BM25 full-text search.
    
    Fusion: Reciprocal Rank Fusion (RRF)
        score(d) = sum_i(1 / (k + rank_i(d)))
    where k=60 (constant), rank_i is rank of document d in retriever i.
    """
    
    def __init__(
        self,
        db_pool: asyncpg.Pool,
        embeddings: BedrockEmbeddings,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        top_k: int = 20,
        rerank_top_k: int = 5,
        rrf_k: int = 60
    ):
        self.db_pool = db_pool
        self.embeddings = embeddings
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k
    
    async def retrieve(self, query: str) -> List[RetrievedDocument]:
        """Perform hybrid retrieval."""
        
        # 1. Vector search
        query_embedding = await self.embeddings.embed_query(query)
        vector_results = await self._vector_search(query_embedding)
        
        # 2. BM25 full-text search
        bm25_results = await self._bm25_search(query)
        
        # 3. Fuse results using RRF
        fused_results = self._reciprocal_rank_fusion(
            vector_results, bm25_results
        )
        
        # 4. Return top-k
        return fused_results[:self.top_k]
    
    async def _vector_search(
        self, query_embedding: np.ndarray
    ) -> List[RetrievedDocument]:
        """Dense vector similarity search using pgvector."""
        
        query = """
            SELECT 
                id, text, metadata, source_document_id,
                1 - (embedding <=> $1::vector) AS score
            FROM documents
            WHERE version = (SELECT MAX(version) FROM documents)
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                query_embedding.tolist(),
                self.top_k * 2  # Retrieve more for fusion
            )
        
        return [
            RetrievedDocument(
                id=str(row["id"]),
                text=row["text"],
                score=float(row["score"]),
                metadata=row["metadata"],
                source_document_id=row["source_document_id"]
            )
            for row in rows
        ]
    
    async def _bm25_search(self, query: str) -> List[RetrievedDocument]:
        """BM25-style full-text search using Postgres tsvector."""
        
        query_sql = """
            SELECT 
                id, text, metadata, source_document_id,
                ts_rank_cd(tsv, plainto_tsquery('english', $1)) AS score
            FROM documents
            WHERE 
                version = (SELECT MAX(version) FROM documents)
                AND tsv @@ plainto_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $2
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query_sql, query, self.top_k * 2)
        
        return [
            RetrievedDocument(
                id=str(row["id"]),
                text=row["text"],
                score=float(row["score"]),
                metadata=row["metadata"],
                source_document_id=row["source_document_id"]
            )
            for row in rows
        ]
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievedDocument],
        bm25_results: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Fuse results using Reciprocal Rank Fusion.
        
        RRF formula: score(d) = sum_i(1 / (k + rank_i(d)))
        where k=60, rank_i is 1-indexed rank in retriever i.
        """
        doc_scores = {}
        
        # Add vector scores
        for rank, doc in enumerate(vector_results, start=1):
            rrf_score = self.vector_weight / (self.rrf_k + rank)
            if doc.id not in doc_scores:
                doc_scores[doc.id] = {"doc": doc, "score": 0.0}
            doc_scores[doc.id]["score"] += rrf_score
        
        # Add BM25 scores
        for rank, doc in enumerate(bm25_results, start=1):
            rrf_score = self.bm25_weight / (self.rrf_k + rank)
            if doc.id not in doc_scores:
                doc_scores[doc.id] = {"doc": doc, "score": 0.0}
            doc_scores[doc.id]["score"] += rrf_score
        
        # Sort by fused score
        fused = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Update scores in documents
        result = []
        for item in fused:
            doc = item["doc"]
            doc.score = item["score"]
            result.append(doc)
        
        return result
```

### 4.3 Generate Node (Bedrock LLM)

```python
# graph/nodes/generate.py
from typing import Dict
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from graph.state import RAGState


SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question based on the provided context.

Guidelines:
- Use ONLY information from the context to answer
- If the context doesn't contain enough information, say so clearly
- Cite sources when making specific claims
- Be concise and direct

Context:
{context}
"""


async def generate_node(state: RAGState) -> Dict:
    """
    Generate answer using Bedrock LLM with retrieved context.
    
    Uses Claude 3.5 Sonnet by default.
    """
    query = state["query"]
    reranked_docs = state.get("reranked_documents", [])
    
    # Format context from reranked documents
    context = "\n\n".join([
        f"[Source {i+1}] {doc['text']}"
        for i, doc in enumerate(reranked_docs)
    ])
    
    # Initialize Bedrock LLM
    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        model_kwargs={
            "temperature": 0.0,
            "max_tokens": 2048,
        }
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{query}")
    ])
    
    # Generate answer
    chain = prompt | llm
    response = await chain.ainvoke({
        "context": context,
        "query": query
    })
    
    return {
        "answer": response.content,
        "context": context,
        "metadata": {
            **state.get("metadata", {}),
            "model": "claude-3.5-sonnet",
            "prompt_tokens": response.response_metadata.get("usage", {}).get("input_tokens"),
            "completion_tokens": response.response_metadata.get("usage", {}).get("output_tokens"),
        }
    }
```

### 4.4 Embeddings Wrapper

```python
# rag/embeddings.py
import boto3
import json
import numpy as np
from typing import List
from functools import lru_cache


class BedrockEmbeddings:
    """
    Wrapper for AWS Bedrock embeddings.
    
    Uses Amazon Titan Embeddings v2 (1024 dimensions).
    Supports batch embedding for efficiency.
    """
    
    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        dimensions: int = 1024,
        normalize: bool = True
    ):
        self.model_id = model_id
        self.dimensions = dimensions
        self.normalize = normalize
        self.client = boto3.client("bedrock-runtime", region_name=region)
    
    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query."""
        return (await self.embed_documents([text]))[0]
    
    async def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed multiple documents.
        
        Note: Bedrock Titan supports batch embedding up to 128 texts.
        For larger batches, we chunk and process sequentially.
        """
        batch_size = 128
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    async def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a single batch via Bedrock."""
        embeddings = []
        
        for text in texts:
            body = json.dumps({
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": self.normalize
            })
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response["body"].read())
            embedding = np.array(response_body["embedding"], dtype=np.float32)
            embeddings.append(embedding)
        
        return embeddings
```

---

## 5. Evaluation Framework

### 5.1 DeepEval + RAGAS Metrics

```python
# eval/metrics.py
from typing import List, Dict
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate


class RAGEvaluator:
    """
    RAG evaluation using DeepEval with RAGAS-style metrics.
    
    Metrics:
    - Faithfulness: Is the answer grounded in the context?
    - Context Relevancy: Are retrieved docs relevant to query?
    - Answer Relevancy: Is the answer relevant to the query?
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.faithfulness_metric = FaithfulnessMetric(
            threshold=0.7,
            model=model,
            include_reason=True
        )
        self.context_relevancy_metric = ContextualRelevancyMetric(
            threshold=0.7,
            model=model,
            include_reason=True
        )
        self.answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=model,
            include_reason=True
        )
    
    async def evaluate_single(
        self,
        query: str,
        answer: str,
        context: List[str],
        expected_answer: str = None
    ) -> Dict[str, float]:
        """Evaluate a single RAG output."""
        
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=expected_answer,
            retrieval_context=context
        )
        
        # Compute metrics
        await self.faithfulness_metric.a_measure(test_case)
        await self.context_relevancy_metric.a_measure(test_case)
        await self.answer_relevancy_metric.a_measure(test_case)
        
        return {
            "faithfulness": self.faithfulness_metric.score,
            "context_relevancy": self.context_relevancy_metric.score,
            "answer_relevancy": self.answer_relevancy_metric.score,
            "faithfulness_reason": self.faithfulness_metric.reason,
            "context_relevancy_reason": self.context_relevancy_metric.reason,
            "answer_relevancy_reason": self.answer_relevancy_metric.reason,
        }
    
    async def evaluate_batch(
        self,
        test_cases: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate multiple test cases and return aggregate metrics.
        
        test_cases: List of dicts with keys:
            - query
            - answer
            - context (list of strings)
            - expected_answer (optional)
        """
        results = []
        
        for case in test_cases:
            metrics = await self.evaluate_single(
                query=case["query"],
                answer=case["answer"],
                context=case["context"],
                expected_answer=case.get("expected_answer")
            )
            results.append(metrics)
        
        # Aggregate
        aggregate = {
            "faithfulness_mean": np.mean([r["faithfulness"] for r in results]),
            "faithfulness_std": np.std([r["faithfulness"] for r in results]),
            "context_relevancy_mean": np.mean([r["context_relevancy"] for r in results]),
            "context_relevancy_std": np.std([r["context_relevancy"] for r in results]),
            "answer_relevancy_mean": np.mean([r["answer_relevancy"] for r in results]),
            "answer_relevancy_std": np.std([r["answer_relevancy"] for r in results]),
            "num_examples": len(results),
            "individual_results": results
        }
        
        return aggregate
```

### 5.2 Evaluation Runner

```python
# eval/runner.py
import asyncio
import json
from pathlib import Path
from typing import List, Dict
from uuid import uuid4
from datetime import datetime

from eval.metrics import RAGEvaluator
from eval.datasets import load_eval_dataset
from eval.thresholds import QUALITY_THRESHOLDS
from graph.rag_graph import rag_graph
from db.repositories.evaluation_results import EvaluationResultsRepository


class EvaluationRunner:
    """
    Run RAG evaluation pipeline.
    
    Steps:
    1. Load evaluation dataset
    2. Run RAG pipeline for each example
    3. Compute metrics using DeepEval
    4. Compare against thresholds
    5. Store results in DB and S3
    """
    
    def __init__(
        self,
        evaluator: RAGEvaluator,
        results_repo: EvaluationResultsRepository,
        s3_client
    ):
        self.evaluator = evaluator
        self.results_repo = results_repo
        self.s3_client = s3_client
    
    async def run_evaluation(
        self,
        dataset_path: str,
        model_version: str,
        num_samples: int = None
    ) -> Dict:
        """
        Run full evaluation pipeline.
        
        Returns:
            Dict with metrics, pass/fail status, and details.
        """
        eval_run_id = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        
        # Load dataset
        eval_examples = load_eval_dataset(dataset_path)
        if num_samples:
            eval_examples = eval_examples[:num_samples]
        
        print(f"Running evaluation {eval_run_id} on {len(eval_examples)} examples...")
        
        # Run RAG pipeline for each example
        test_cases = []
        for example in eval_examples:
            # Invoke RAG graph
            result = await rag_graph.ainvoke({
                "query": example["query"],
                "messages": [],
                "metadata": {"eval_run_id": eval_run_id}
            })
            
            test_cases.append({
                "query": example["query"],
                "answer": result["answer"],
                "context": [doc["text"] for doc in result.get("reranked_documents", [])],
                "expected_answer": example.get("expected_answer"),
                "ground_truth_docs": example.get("ground_truth_docs", [])
            })
        
        # Evaluate
        metrics = await self.evaluator.evaluate_batch(test_cases)
        
        # Check thresholds
        passed = self._check_thresholds(metrics)
        
        # Store results
        await self.results_repo.store_result(
            eval_run_id=eval_run_id,
            dataset_version=Path(dataset_path).stem,
            model_version=model_version,
            metrics=metrics,
            num_examples=len(eval_examples),
            passed=passed
        )
        
        # Upload detailed results to S3
        await self._upload_to_s3(eval_run_id, metrics)
        
        return {
            "eval_run_id": eval_run_id,
            "metrics": metrics,
            "num_examples": len(eval_examples),
            "passed": passed,
            "thresholds": QUALITY_THRESHOLDS
        }
    
    def _check_thresholds(self, metrics: Dict) -> bool:
        """Check if metrics meet quality thresholds."""
        return (
            metrics["faithfulness_mean"] >= QUALITY_THRESHOLDS["faithfulness"]
            and metrics["context_relevancy_mean"] >= QUALITY_THRESHOLDS["context_relevancy"]
            and metrics["answer_relevancy_mean"] >= QUALITY_THRESHOLDS["answer_relevancy"]
        )
    
    async def _upload_to_s3(self, eval_run_id: str, metrics: Dict):
        """Upload detailed results to S3."""
        s3_key = f"eval-results/{eval_run_id}/metrics.json"
        self.s3_client.put_object(
            Bucket="rag-system-artifacts",
            Key=s3_key,
            Body=json.dumps(metrics, indent=2)
        )
```

```python
# eval/thresholds.py
"""Quality gate thresholds for CI/CD."""

QUALITY_THRESHOLDS = {
    "faithfulness": 0.75,          # Answers must be grounded
    "context_relevancy": 0.70,     # Retrieved docs must be relevant
    "answer_relevancy": 0.75,      # Answers must address the query
}

# Warn thresholds (log warnings but don't fail)
WARN_THRESHOLDS = {
    "faithfulness": 0.80,
    "context_relevancy": 0.75,
    "answer_relevancy": 0.80,
}
```

### 5.3 Evaluation Dataset Format

```jsonl
# eval/datasets/eval_v1.jsonl
{"query": "What is the refund policy for orders over $100?", "expected_answer": "Orders over $100 are eligible for full refunds within 30 days of purchase.", "ground_truth_docs": ["doc_id_123"], "metadata": {"category": "policy", "difficulty": "easy"}}
{"query": "How do I integrate the payment API with React?", "expected_answer": "Install the SDK via npm, initialize with your API key, and use the <PaymentForm> component.", "ground_truth_docs": ["doc_id_456", "doc_id_789"], "metadata": {"category": "technical", "difficulty": "medium"}}
```

---

## 6. Observability & Monitoring

### 6.1 Tracing Integration

```python
# observability/tracing.py
import os
from typing import Dict, Any, Optional
from langsmith import Client
from langsmith.run_helpers import traceable
from contextlib import asynccontextmanager


class ObservabilityManager:
    """
    Manage observability across LangSmith, CloudWatch, and custom logging.
    
    For production:
    - Use LangSmith for detailed LLM tracing
    - Use CloudWatch for metrics and logs
    - Use custom DB tracing for analytics
    """
    
    def __init__(self):
        self.langsmith_client = None
        if os.getenv("LANGSMITH_API_KEY"):
            self.langsmith_client = Client()
    
    @asynccontextmanager
    async def trace_request(
        self,
        request_id: str,
        operation: str,
        metadata: Dict[str, Any]
    ):
        """
        Context manager for tracing a request.
        
        Usage:
            async with obs_manager.trace_request(req_id, "chat", {...}) as tracer:
                result = await rag_graph.ainvoke(...)
                tracer.log_result(result)
        """
        tracer = RequestTracer(
            request_id=request_id,
            operation=operation,
            metadata=metadata,
            langsmith_client=self.langsmith_client
        )
        
        try:
            yield tracer
        finally:
            await tracer.finalize()


class RequestTracer:
    """Trace a single request through the system."""
    
    def __init__(
        self,
        request_id: str,
        operation: str,
        metadata: Dict[str, Any],
        langsmith_client: Optional[Client]
    ):
        self.request_id = request_id
        self.operation = operation
        self.metadata = metadata
        self.langsmith_client = langsmith_client
        self.start_time = time.time()
        self.result = None
        self.error = None
    
    def log_result(self, result: Dict[str, Any]):
        """Log the result of the operation."""
        self.result = result
    
    def log_error(self, error: Exception):
        """Log an error."""
        self.error = error
    
    async def finalize(self):
        """Finalize the trace and send to backends."""
        latency_ms = int((time.time() - self.start_time) * 1000)
        
        # 1. Log to CloudWatch
        logger.info(
            "Request completed",
            extra={
                "request_id": self.request_id,
                "operation": self.operation,
                "latency_ms": latency_ms,
                "metadata": self.metadata,
                "error": str(self.error) if self.error else None
            }
        )
        
        # 2. Store in database for analytics
        # (handled by repositories)
        
        # 3. Send to LangSmith if configured
        if self.langsmith_client and self.result:
            # LangSmith auto-captures via @traceable decorator
            pass
```

### 6.2 Structured Logging

```python
# observability/logging.py
import logging
import json
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging to CloudWatch.
    
    All logs include:
    - timestamp
    - level
    - message
    - request_id (if available)
    - extra fields
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info"
            ]:
                log_data[key] = value
        
        return json.dumps(log_data)


def setup_logging():
    """Setup structured logging for the application."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    
    return root_logger
```

### 6.3 CloudWatch Metrics

```python
# observability/metrics.py
import boto3
from typing import Dict, List
from datetime import datetime


class MetricsPublisher:
    """
    Publish custom metrics to CloudWatch.
    
    Key metrics:
    - Request latency (p50, p90, p99)
    - Request count
    - Error rate
    - Token usage
    - Cost per request
    - Eval scores (faithfulness, relevancy)
    """
    
    def __init__(self, namespace: str = "RAGSystem"):
        self.cloudwatch = boto3.client("cloudwatch")
        self.namespace = namespace
    
    async def publish_request_metrics(
        self,
        latency_ms: int,
        tokens: int,
        cost_usd: float,
        model: str,
        success: bool
    ):
        """Publish metrics for a single request."""
        
        metric_data = [
            {
                "MetricName": "RequestLatency",
                "Value": latency_ms,
                "Unit": "Milliseconds",
                "Timestamp": datetime.utcnow(),
                "Dimensions": [
                    {"Name": "Model", "Value": model}
                ]
            },
            {
                "MetricName": "TokenUsage",
                "Value": tokens,
                "Unit": "Count",
                "Timestamp": datetime.utcnow(),
                "Dimensions": [
                    {"Name": "Model", "Value": model}
                ]
            },
            {
                "MetricName": "RequestCost",
                "Value": cost_usd,
                "Unit": "None",
                "Timestamp": datetime.utcnow(),
                "Dimensions": [
                    {"Name": "Model", "Value": model}
                ]
            },
            {
                "MetricName": "RequestCount",
                "Value": 1,
                "Unit": "Count",
                "Timestamp": datetime.utcnow(),
                "Dimensions": [
                    {"Name": "Model", "Value": model},
                    {"Name": "Status", "Value": "Success" if success else "Error"}
                ]
            }
        ]
        
        self.cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=metric_data
        )
    
    async def publish_eval_metrics(
        self,
        eval_run_id: str,
        metrics: Dict[str, float]
    ):
        """Publish evaluation metrics."""
        
        metric_data = [
            {
                "MetricName": "Faithfulness",
                "Value": metrics["faithfulness_mean"],
                "Unit": "None",
                "Timestamp": datetime.utcnow(),
                "Dimensions": [
                    {"Name": "EvalRunId", "Value": eval_run_id}
                ]
            },
            {
                "MetricName": "ContextRelevancy",
                "Value": metrics["context_relevancy_mean"],
                "Unit": "None",
                "Timestamp": datetime.utcnow(),
                "Dimensions": [
                    {"Name": "EvalRunId", "Value": eval_run_id}
                ]
            },
            {
                "MetricName": "AnswerRelevancy",
                "Value": metrics["answer_relevancy_mean"],
                "Unit": "None",
                "Timestamp": datetime.utcnow(),
                "Dimensions": [
                    {"Name": "EvalRunId", "Value": eval_run_id}
                ]
            }
        ]
        
        self.cloudwatch.put_metric_data(
            Namespace=self.namespace,
            MetricData=metric_data
        )
```

---

## 7. Feedback Loops

### 7.1 Feedback Collection API

```python
# app/routers/feedback.py
from fastapi import APIRouter, Depends, HTTPException
from app.models import FeedbackRequest, FeedbackResponse
from db.repositories.feedback import FeedbackRepository
from db.repositories.traces import TracesRepository


router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    feedback_repo: FeedbackRepository = Depends(),
    traces_repo: TracesRepository = Depends()
):
    """
    Submit user feedback on a response.
    
    Links feedback to the original request trace.
    """
    # Verify trace exists
    trace = await traces_repo.get_by_request_id(request.request_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Store feedback
    feedback_id = await feedback_repo.create_feedback(
        trace_id=trace.id,
        user_id=request.metadata.get("user_id"),
        rating=request.rating,
        comment=request.comment,
        metadata=request.metadata
    )
    
    return FeedbackResponse(
        feedback_id=feedback_id,
        status="success"
    )
```

### 7.2 Feedback → Evaluation Dataset

```python
# feedback/processor.py
import json
from typing import List, Dict
from datetime import datetime, timedelta
from db.repositories.feedback import FeedbackRepository
from db.repositories.traces import TracesRepository


class FeedbackProcessor:
    """
    Process feedback to create new evaluation examples.
    
    Strategy:
    1. Query low-rated interactions (thumbs down, low eval scores)
    2. Sample diverse examples (avoid duplication)
    3. Format as evaluation examples
    4. Append to evaluation dataset on S3
    """
    
    def __init__(
        self,
        feedback_repo: FeedbackRepository,
        traces_repo: TracesRepository,
        s3_client
    ):
        self.feedback_repo = feedback_repo
        self.traces_repo = traces_repo
        self.s3_client = s3_client
    
    async def export_negative_feedback_to_eval_set(
        self,
        lookback_days: int = 7,
        min_samples: int = 10,
        max_samples: int = 100
    ) -> Dict:
        """
        Export negative feedback as new evaluation examples.
        
        Returns:
            Dict with stats on exported examples.
        """
        since = datetime.utcnow() - timedelta(days=lookback_days)
        
        # Query negative feedback
        negative_feedback = await self.feedback_repo.get_negative_feedback(
            since=since,
            limit=max_samples
        )
        
        if len(negative_feedback) < min_samples:
            return {
                "status": "skipped",
                "reason": f"Insufficient samples ({len(negative_feedback)} < {min_samples})"
            }
        
        # Convert to evaluation examples
        eval_examples = []
        for feedback in negative_feedback:
            trace = await self.traces_repo.get_by_id(feedback.trace_id)
            
            # Human review needed: add placeholder for expected answer
            eval_example = {
                "query": trace.query,
                "actual_answer": trace.response,
                "retrieved_docs": trace.retrieved_doc_ids,
                "expected_answer": None,  # Requires human annotation
                "ground_truth_docs": None,  # Requires human annotation
                "metadata": {
                    "source": "negative_feedback",
                    "feedback_id": str(feedback.id),
                    "trace_id": str(trace.id),
                    "rating": feedback.rating,
                    "comment": feedback.comment,
                    "created_at": feedback.created_at.isoformat()
                }
            }
            eval_examples.append(eval_example)
        
        # Upload to S3 for human review
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_key = f"eval-datasets/feedback-derived/unannotated_{timestamp}.jsonl"
        
        lines = [json.dumps(ex) for ex in eval_examples]
        self.s3_client.put_object(
            Bucket="rag-system-artifacts",
            Key=s3_key,
            Body="\n".join(lines)
        )
        
        return {
            "status": "success",
            "num_examples": len(eval_examples),
            "s3_key": s3_key,
            "message": "Examples exported for human annotation"
        }
```

### 7.3 Continuous Improvement Loop

```
┌─────────────────────────────────────────────────────────────┐
│                  Continuous Improvement Loop                 │
└─────────────────────────────────────────────────────────────┘

1. Production Traffic
   └─> User queries → RAG system → Responses
   
2. Feedback Collection
   └─> Users rate responses (thumbs up/down, comments)
   
3. Trace Analysis
   └─> Low-rated responses flagged
   └─> Automatic eval metrics computed
   
4. Example Generation
   └─> Negative feedback → Unannotated eval examples
   └─> Human reviewers add expected answers
   
5. Evaluation Dataset Update
   └─> Annotated examples added to eval set
   └─> Dataset versioned in S3
   
6. Regression Testing
   └─> CI runs eval on every PR
   └─> New examples become regression tests
   
7. Model/System Improvements
   └─> Adjust prompts, retrieval, chunking, etc.
   └─> Ensure fixes don't break existing cases
   
8. Deployment
   └─> Changes pass eval thresholds → Deploy
   
[Loop back to 1]
```

---

## 8. AWS Infrastructure

### 8.1 Infrastructure Components

```
AWS Resources:
├── Compute
│   ├── ECS Fargate Cluster
│   │   ├── Service: rag-api (FastAPI)
│   │   ├── Task Definition: 2 vCPU, 4GB RAM
│   │   └── Auto-scaling: 2-10 tasks
│   └── Lambda (optional)
│       └── S3 event trigger → Ingestion
│
├── Storage
│   ├── RDS Postgres (db.r5.large)
│   │   ├── Multi-AZ for HA
│   │   ├── pgvector extension
│   │   └── Automated backups
│   └── S3 Buckets
│       ├── rag-system-documents
│       ├── rag-system-artifacts
│       └── rag-system-eval-datasets
│
├── Networking
│   ├── VPC (10.0.0.0/16)
│   ├── Public Subnets (ALB)
│   ├── Private Subnets (ECS, RDS)
│   └── NAT Gateway
│
├── Load Balancing
│   └── Application Load Balancer
│       ├── HTTPS (ACM certificate)
│       ├── Health checks: /health
│       └── Target: ECS Service
│
├── Security
│   ├── IAM Roles
│   │   ├── ECS Task Role (Bedrock, S3, Secrets)
│   │   └── ECS Execution Role (ECR, CloudWatch)
│   ├── Security Groups
│   │   ├── ALB: 443 from 0.0.0.0/0
│   │   ├── ECS: 8000 from ALB
│   │   └── RDS: 5432 from ECS
│   └── Secrets Manager
│       ├── db-password
│       ├── langsmith-api-key
│       └── api-keys
│
└── Monitoring
    ├── CloudWatch Logs
    │   ├── /ecs/rag-api
    │   └── /aws/rds/instance/rag-db
    ├── CloudWatch Metrics
    │   └── Custom namespace: RAGSystem
    └── CloudWatch Alarms
        ├── High error rate
        ├── High latency (p99 > 5s)
        └── Low eval scores
```

### 8.2 Terraform Example

```hcl
# infra/terraform/main.tf

terraform {
  required_version = ">= 1.5"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "rag-system-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  default = "us-east-1"
}

variable "environment" {
  default = "production"
}

# VPC
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "rag-system-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  single_nat_gateway = false  # Use one NAT per AZ for HA
  
  tags = {
    Environment = var.environment
    Project     = "rag-system"
  }
}

# RDS Postgres
resource "aws_db_instance" "postgres" {
  identifier = "rag-system-db"
  
  engine         = "postgres"
  engine_version = "16.1"
  instance_class = "db.r5.large"
  
  allocated_storage     = 100
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "ragdb"
  username = "rag_admin"
  password = aws_secretsmanager_secret_version.db_password.secret_string
  
  multi_az               = true
  publicly_accessible    = false
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"
  
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  
  tags = {
    Environment = var.environment
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "rag-system-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "api" {
  family                   = "rag-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "2048"
  memory                   = "4096"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn
  
  container_definitions = jsonencode([
    {
      name  = "api"
      image = "${aws_ecr_repository.api.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "DB_HOST"
          value = aws_db_instance.postgres.address
        }
      ]
      
      secrets = [
        {
          name      = "DB_PASSWORD"
          valueFrom = aws_secretsmanager_secret.db_password.arn
        },
        {
          name      = "LANGSMITH_API_KEY"
          valueFrom = aws_secretsmanager_secret.langsmith_key.arn
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/rag-api"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "api"
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# ECS Service
resource "aws_ecs_service" "api" {
  name            = "rag-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = module.vpc.private_subnets
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
  
  depends_on = [aws_lb_listener.https]
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "rag-system-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets
}

resource "aws_lb_target_group" "api" {
  name        = "rag-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  target_type = "ip"
  
  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 10
    timeout             = 5
    interval            = 30
  }
}

# S3 Buckets
resource "aws_s3_bucket" "documents" {
  bucket = "rag-system-documents-${var.environment}"
  
  tags = {
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# IAM Role for ECS Task
resource "aws_iam_role" "ecs_task_role" {
  name = "rag-ecs-task-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "ecs_task_bedrock" {
  role = aws_iam_role.ecs_task_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = [
          "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-sonnet-*",
          "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${aws_s3_bucket.documents.arn}/*",
          aws_s3_bucket.documents.arn
        ]
      }
    ]
  })
}
```

### 8.3 Dockerfile

```dockerfile
# infra/docker/Dockerfile

FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Copy dependency files
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main

# Runtime stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 9. CI/CD Pipeline

### 9.1 GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml

name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: rag-api
  ECS_CLUSTER: rag-system-cluster
  ECS_SERVICE: rag-api
  PYTHON_VERSION: "3.11"

jobs:
  lint-and-test:
    name: Lint and Test
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Cache Poetry
        uses: actions/cache@v3
        with:
          path: |
            ~/.cache/pypoetry
            .venv
          key: poetry-${{ runner.os }}-${{ hashFiles('poetry.lock') }}
      
      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      
      - name: Install dependencies
        run: poetry install --no-interaction
      
      - name: Lint with ruff
        run: |
          poetry run ruff check .
          poetry run ruff format --check .
      
      - name: Type check with mypy
        run: poetry run mypy app/ graph/ rag/ eval/
      
      - name: Run unit tests
        run: poetry run pytest tests/unit/ -v --cov=app --cov=rag --cov=graph
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: lint-and-test
    
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      
      - name: Install dependencies
        run: poetry install --no-interaction
      
      - name: Run migrations
        env:
          DB_HOST: localhost
          DB_PORT: 5432
          DB_NAME: test_db
          DB_USER: test_user
          DB_PASSWORD: test_password
        run: |
          poetry run alembic upgrade head
      
      - name: Run integration tests
        env:
          DB_HOST: localhost
          DB_PORT: 5432
          DB_NAME: test_db
          DB_USER: test_user
          DB_PASSWORD: test_password
        run: poetry run pytest tests/integration/ -v

  evaluation:
    name: Run Evaluation Suite
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.event_name == 'pull_request'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      
      - name: Install dependencies
        run: poetry install --no-interaction
      
      - name: Download evaluation dataset from S3
        run: |
          aws s3 cp s3://rag-system-eval-datasets/eval_v1.jsonl eval/datasets/
      
      - name: Run evaluation
        env:
          DB_HOST: ${{ secrets.STAGING_DB_HOST }}
          DB_PASSWORD: ${{ secrets.STAGING_DB_PASSWORD }}
          BEDROCK_REGION: ${{ env.AWS_REGION }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}  # For DeepEval metrics
        run: |
          poetry run python scripts/run_evaluation.py \
            --dataset eval/datasets/eval_v1.jsonl \
            --model-version ${{ github.sha }} \
            --output eval_results.json
      
      - name: Check evaluation thresholds
        run: |
          poetry run python scripts/check_eval_thresholds.py \
            --results eval_results.json \
            --fail-on-regression
      
      - name: Upload evaluation results
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-results
          path: eval_results.json
      
      - name: Store results in S3
        run: |
          aws s3 cp eval_results.json \
            s3://rag-system-artifacts/eval-results/${{ github.sha }}/
      
      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('eval_results.json', 'utf8'));
            
            const body = `
            ## 📊 Evaluation Results
            
            **Model Version:** \`${{ github.sha }}\`
            
            ### Metrics
            - **Faithfulness:** ${results.metrics.faithfulness_mean.toFixed(3)} (threshold: 0.75)
            - **Context Relevancy:** ${results.metrics.context_relevancy_mean.toFixed(3)} (threshold: 0.70)
            - **Answer Relevancy:** ${results.metrics.answer_relevancy_mean.toFixed(3)} (threshold: 0.75)
            
            **Status:** ${results.passed ? '✅ PASSED' : '❌ FAILED'}
            
            [View detailed results](https://s3.console.aws.amazon.com/s3/object/rag-system-artifacts/eval-results/${{ github.sha }}/eval_results.json)
            `;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });

  build-and-push:
    name: Build and Push Docker Image
    runs-on: ubuntu-latest
    needs: [integration-tests, evaluation]
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
    
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}
          tags: |
            type=sha,prefix={{branch}}-
            type=ref,event=branch
            type=semver,pattern={{version}}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: infra/docker/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    name: Deploy to ECS
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Update ECS task definition
        id: task-def
        run: |
          # Get current task definition
          TASK_DEF=$(aws ecs describe-task-definition --task-definition rag-api --region ${{ env.AWS_REGION }})
          
          # Update image
          NEW_TASK_DEF=$(echo $TASK_DEF | jq --arg IMAGE "${{ needs.build-and-push.outputs.image-tag }}" \
            '.taskDefinition | .containerDefinitions[0].image = $IMAGE | del(.taskDefinitionArn, .revision, .status, .requiresAttributes, .compatibilities, .registeredAt, .registeredBy)')
          
          # Register new task definition
          NEW_TASK_ARN=$(aws ecs register-task-definition --region ${{ env.AWS_REGION }} --cli-input-json "$NEW_TASK_DEF" | jq -r '.taskDefinition.taskDefinitionArn')
          
          echo "task-def-arn=$NEW_TASK_ARN" >> $GITHUB_OUTPUT
      
      - name: Deploy to ECS (Blue/Green)
        run: |
          aws ecs update-service \
            --cluster ${{ env.ECS_CLUSTER }} \
            --service ${{ env.ECS_SERVICE }} \
            --task-definition ${{ steps.task-def.outputs.task-def-arn }} \
            --force-new-deployment \
            --region ${{ env.AWS_REGION }}
      
      - name: Wait for deployment
        run: |
          aws ecs wait services-stable \
            --cluster ${{ env.ECS_CLUSTER }} \
            --services ${{ env.ECS_SERVICE }} \
            --region ${{ env.AWS_REGION }}
      
      - name: Verify deployment
        run: |
          # Check service health
          SERVICE_STATUS=$(aws ecs describe-services \
            --cluster ${{ env.ECS_CLUSTER }} \
            --services ${{ env.ECS_SERVICE }} \
            --region ${{ env.AWS_REGION }} \
            | jq -r '.services[0].deployments[0].rolloutState')
          
          if [ "$SERVICE_STATUS" != "COMPLETED" ]; then
            echo "Deployment failed: $SERVICE_STATUS"
            exit 1
          fi
      
      - name: Notify Slack
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            Deployment to production ${{ job.status }}
            Image: ${{ needs.build-and-push.outputs.image-tag }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### 9.2 Evaluation Threshold Check Script

```python
# scripts/check_eval_thresholds.py
import argparse
import json
import sys
from eval.thresholds import QUALITY_THRESHOLDS, WARN_THRESHOLDS


def check_thresholds(results_path: str, fail_on_regression: bool = True):
    """
    Check if evaluation results meet quality thresholds.
    
    Exit codes:
    - 0: All thresholds met
    - 1: Hard threshold violation (fail)
    - 2: Warn threshold violation (warning only)
    """
    with open(results_path) as f:
        results = json.load(f)
    
    metrics = results["metrics"]
    
    # Check hard thresholds
    violations = []
    for metric, threshold in QUALITY_THRESHOLDS.items():
        actual = metrics.get(f"{metric}_mean", 0.0)
        if actual < threshold:
            violations.append(f"{metric}: {actual:.3f} < {threshold}")
    
    # Check warn thresholds
    warnings = []
    for metric, threshold in WARN_THRESHOLDS.items():
        actual = metrics.get(f"{metric}_mean", 0.0)
        if actual < threshold:
            warnings.append(f"{metric}: {actual:.3f} < {threshold}")
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    for metric in ["faithfulness", "context_relevancy", "answer_relevancy"]:
        mean = metrics.get(f"{metric}_mean", 0.0)
        std = metrics.get(f"{metric}_std", 0.0)
        threshold = QUALITY_THRESHOLDS[metric]
        warn_threshold = WARN_THRESHOLDS[metric]
        
        status = "✅" if mean >= threshold else "❌"
        print(f"{status} {metric.replace('_', ' ').title():25} {mean:.3f} ± {std:.3f} (threshold: {threshold}, warn: {warn_threshold})")
    
    print("=" * 60)
    
    if violations:
        print(f"\n❌ THRESHOLD VIOLATIONS ({len(violations)}):")
        for v in violations:
            print(f"  - {v}")
        
        if fail_on_regression:
            print("\n💥 Failing CI due to threshold violations.")
            sys.exit(1)
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
        print("\n⚠️  Consider investigating these metrics.")
        sys.exit(2)
    
    print("\n✅ All thresholds met!")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to eval results JSON")
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()
    
    check_thresholds(args.results, args.fail_on_regression)
```

---

## 10. Scalability & Future Extensions

### 10.1 Current Architecture Scalability

**Horizontal Scaling**:
- ECS Auto-scaling: 2-10 tasks based on CPU/memory/request count
- RDS read replicas for read-heavy workloads
- pgvector HNSW index for sub-linear search at scale

**Vertical Scaling**:
- Upgrade RDS instance class (db.r5.large → db.r5.xlarge)
- Increase ECS task resources (2 vCPU → 4 vCPU)

**Bottlenecks**:
1. **Database**: 
   - Vector similarity search is CPU-intensive
   - Solution: Add read replicas, partition by version/date
2. **Embeddings**: 
   - Bedrock has rate limits
   - Solution: Batch embedding requests, implement retry with backoff
3. **Generation**:
   - Bedrock invocations are sequential
   - Solution: Implement streaming responses, cache common queries

### 10.2 Multi-Tenancy

```python
# Multi-tenant schema modification
ALTER TABLE documents ADD COLUMN tenant_id TEXT NOT NULL;
CREATE INDEX idx_documents_tenant ON documents(tenant_id);

ALTER TABLE traces ADD COLUMN tenant_id TEXT NOT NULL;
CREATE INDEX idx_traces_tenant ON traces(tenant_id);

# Row-level security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON documents
    USING (tenant_id = current_setting('app.current_tenant'));
```

**Tenant isolation strategies**:
1. **Database-level**: Separate RDS instances per tenant (strong isolation, high cost)
2. **Schema-level**: Separate schemas per tenant (good isolation, moderate cost)
3. **Row-level**: Single schema with tenant_id (weak isolation, low cost, scale)

### 10.3 Advanced RAG Features

**Query Decomposition**:
```python
# graph/nodes/decompose.py
async def decompose_node(state: RAGState) -> Dict:
    """
    Decompose complex queries into sub-queries.
    
    e.g., "Compare pricing of Pro vs Enterprise plans"
    →  ["What is Pro plan pricing?", "What is Enterprise plan pricing?"]
    """
    llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-...")
    decomposition_prompt = """
    Decompose this query into simpler sub-queries:
    {query}
    
    Return JSON array of sub-queries.
    """
    # ... decomposition logic
    return {"sub_queries": sub_queries}
```

**Self-Correction Loop**:
```python
# graph/nodes/verify.py
async def verify_node(state: RAGState) -> Dict:
    """
    Verify answer quality and trigger re-generation if needed.
    
    Checks:
    - Faithfulness score > 0.7
    - Answer is not "I don't know"
    - Answer addresses all parts of query
    """
    answer = state["answer"]
    
    # Quick faithfulness check
    score = await quick_faithfulness_check(answer, state["context"])
    
    if score < 0.7:
        return {"verification_failed": True, "retry_count": state.get("retry_count", 0) + 1}
    
    return {"verification_failed": False}

# Add conditional edge in graph
workflow.add_conditional_edges(
    "generate",
    lambda state: "verify" if state.get("retry_count", 0) < 2 else "end",
    {
        "verify": "retrieve",  # Re-retrieve with different parameters
        "end": END
    }
)
```

**Tool Calling / Agentic RAG**:
```python
# Add tools for external APIs
tools = [
    {
        "name": "search_confluence",
        "description": "Search internal Confluence wiki",
        "parameters": {...}
    },
    {
        "name": "query_database",
        "description": "Query production database for metrics",
        "parameters": {...}
    }
]

# LLM decides which tools to call
llm_with_tools = ChatBedrock(..., tools=tools)
```

**Multi-modal RAG**:
```python
# Support image documents
# 1. Extract text from images using Textract
# 2. Generate image embeddings using CLIP or Bedrock multimodal
# 3. Store image_url in metadata
# 4. Return images alongside text in context
```

### 10.4 Cost Optimization

**Strategies**:
1. **Caching**:
   - Cache embeddings (deduplicate identical chunks)
   - Cache LLM responses for common queries
   - Use Redis/ElastiCache for hot data

2. **Model Selection**:
   - Use Haiku for simple queries, Sonnet for complex
   - Implement query classifier to route

3. **Batch Processing**:
   - Batch embed operations (128 texts per request)
   - Use Bedrock batch inference for large jobs

4. **Indexing**:
   - Use HNSW index for faster vector search (less compute)
   - Prune old document versions

**Cost Monitoring**:
```python
# Track costs per request
cost_per_request = (
    embedding_tokens * EMBEDDING_COST_PER_1K
    + prompt_tokens * MODEL_INPUT_COST_PER_1K
    + completion_tokens * MODEL_OUTPUT_COST_PER_1K
)

# Publish to CloudWatch for budgeting
await metrics_publisher.publish_cost_metric(cost_per_request, model_name)
```

### 10.5 Governance & Compliance

**PII Detection**:
```python
# Use AWS Comprehend to detect PII before indexing
comprehend = boto3.client("comprehend")
pii_entities = comprehend.detect_pii_entities(Text=chunk_text, LanguageCode="en")

# Redact or skip chunks with PII
if pii_entities["Entities"]:
    chunk_text = redact_pii(chunk_text, pii_entities)
```

**Audit Logging**:
- Log all queries, responses, and feedback
- Retain for compliance (GDPR, SOC2, HIPAA)
- Implement data retention policies

**Content Filtering**:
- Use Bedrock Guardrails to filter harmful content
- Implement custom content policies

---

## Summary

This production-grade RAG system includes:

✅ **Core RAG**: LangChain + LangGraph orchestration, hybrid retrieval (BM25 + vector), AWS Bedrock LLM  
✅ **Storage**: RDS Postgres + pgvector + tsvector, S3 for documents  
✅ **Evaluation**: DeepEval + RAGAS metrics, quality gates in CI, regression testing  
✅ **Observability**: LangSmith tracing, CloudWatch logs/metrics, structured logging  
✅ **Feedback Loops**: User feedback collection, automatic eval dataset generation  
✅ **Infrastructure**: ECS Fargate, ALB, IAM, Terraform IaC  
✅ **CI/CD**: GitHub Actions with linting, testing, evaluation, and deployment  
✅ **Scalability**: Auto-scaling, read replicas, multi-tenancy ready  

**Next Steps**:
1. Implement repo structure and code
2. Set up AWS infrastructure with Terraform
3. Create initial evaluation dataset
4. Deploy to staging and run first eval
5. Iterate on prompts and retrieval based on eval results
6. Deploy to production with monitoring
7. Collect feedback and continuously improve

This architecture is production-ready, observable, and continuously improving through feedback loops and automated evaluation.
