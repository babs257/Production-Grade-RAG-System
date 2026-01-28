# Production RAG System

A production-grade Retrieval-Augmented Generation (RAG) system with comprehensive evaluation, monitoring, and feedback loops, optimized for AWS.

## 🏗️ Architecture

- **Framework**: LangChain + LangGraph for RAG orchestration
- **API**: FastAPI with async support
- **LLM**: AWS Bedrock (Claude 3.5 Sonnet)
- **Embeddings**: AWS Bedrock Titan Embeddings
- **Vector Store**: PostgreSQL with pgvector
- **Search**: Hybrid retrieval (BM25 + dense vectors)
- **Storage**: Amazon S3 for documents and artifacts
- **Infrastructure**: ECS Fargate, RDS, ALB
- **Evaluation**: DeepEval with RAGAS metrics
- **Observability**: LangSmith, CloudWatch

## 📋 Prerequisites

- Python 3.11+
- Docker & Docker Compose
- AWS Account with:
  - Bedrock access (Claude & Titan models)
  - RDS Postgres
  - S3 buckets
  - ECS/Fargate
- Poetry (Python dependency management)

## 🚀 Quick Start

### 1. Local Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd rag-production-system

# Install dependencies
poetry install

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, DB credentials, etc.

# Start services with Docker Compose
docker-compose up -d

# Wait for services to be healthy
docker-compose ps

# Run database migrations
poetry run alembic upgrade head

# Start the API (development mode with hot reload)
poetry run uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 2. Ingest Documents

```bash
# Upload a document to S3
aws s3 cp /path/to/document.pdf s3://rag-system-documents-dev/

# Trigger ingestion via API
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "s3_key": "document.pdf",
    "metadata": {"category": "technical"}
  }'
```

Or use the ingestion script:

```bash
poetry run python scripts/ingest_documents.py \
  --s3-key document.pdf \
  --metadata '{"category": "technical"}'
```

### 3. Query the System

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the refund policy?",
    "user_id": "user123"
  }'
```

### 4. Run Evaluation

```bash
# Prepare evaluation dataset (JSONL format)
# See eval/datasets/eval_v1.jsonl for format

# Run evaluation
poetry run python scripts/run_evaluation.py \
  --dataset eval/datasets/eval_v1.jsonl \
  --model-version v1.0.0

# Check if metrics meet thresholds
poetry run python scripts/check_eval_thresholds.py \
  --results eval_results.json \
  --fail-on-regression
```

## 📁 Project Structure

```
rag-production-system/
├── app/                      # FastAPI application
│   ├── main.py              # Entry point
│   ├── config.py            # Configuration
│   ├── models.py            # Pydantic models
│   ├── dependencies.py      # FastAPI dependencies
│   ├── middleware.py        # Custom middleware
│   └── routers/             # API endpoints
│       ├── chat.py          # /chat
│       ├── ingest.py        # /ingest
│       ├── feedback.py      # /feedback
│       └── eval.py          # /eval
│
├── graph/                   # LangGraph definitions
│   ├── rag_graph.py        # Main RAG graph
│   ├── state.py            # Graph state
│   └── nodes/              # Graph nodes
│       ├── retrieve.py     # Retrieval node
│       ├── rerank.py       # Reranking node
│       └── generate.py     # Generation node
│
├── rag/                    # RAG components
│   ├── chunking.py         # Text chunking
│   ├── embeddings.py       # Bedrock embeddings
│   ├── retrieval.py        # Hybrid retrieval
│   └── ingestion.py        # Document ingestion
│
├── eval/                   # Evaluation framework
│   ├── datasets.py         # Dataset loader
│   ├── metrics.py          # DeepEval metrics
│   ├── runner.py           # Evaluation runner
│   ├── thresholds.py       # Quality gates
│   └── datasets/           # Evaluation data
│
├── db/                     # Database layer
│   ├── connection.py       # Connection pool
│   ├── schema.sql          # Database schema
│   ├── migrations/         # Alembic migrations
│   └── repositories/       # Data access
│
├── observability/          # Monitoring
│   ├── tracing.py          # LangSmith integration
│   ├── logging.py          # Structured logging
│   └── metrics.py          # CloudWatch metrics
│
├── feedback/               # Feedback loops
│   ├── collector.py        # Feedback collection
│   └── processor.py        # Feedback → eval
│
├── infra/                  # Infrastructure
│   ├── terraform/          # IaC
│   └── docker/
│       └── Dockerfile
│
├── tests/                  # Tests
│   ├── unit/
│   ├── integration/
│   └── eval/
│
├── scripts/                # Utility scripts
├── .github/workflows/      # CI/CD
├── docker-compose.yml      # Local development
├── pyproject.toml          # Dependencies
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov=rag --cov=graph --cov-report=html

# Run specific test suite
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest tests/eval/

# View coverage report
open htmlcov/index.html
```

## 🔍 Code Quality

```bash
# Lint with ruff
poetry run ruff check .

# Format with ruff
poetry run ruff format .

# Type check with mypy
poetry run mypy app/ graph/ rag/ eval/

# Run all checks
poetry run ruff check . && \
poetry run ruff format --check . && \
poetry run mypy app/ graph/ rag/ eval/
```

## 🚢 Deployment

### Infrastructure Setup

```bash
cd infra/terraform

# Initialize Terraform
terraform init

# Plan infrastructure changes
terraform plan -out=tfplan

# Apply changes
terraform apply tfplan
```

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) handles:

1. **Lint & Test**: Run linting, type checking, unit tests
2. **Integration Tests**: Spin up Postgres, run integration tests
3. **Evaluation**: Run evaluation suite, check thresholds
4. **Build**: Build Docker image, push to ECR
5. **Deploy**: Deploy to ECS Fargate with blue/green strategy

Every PR runs evaluation and comments with results. Deployments only proceed if metrics meet thresholds.

### Manual Deployment

```bash
# Build Docker image
docker build -f infra/docker/Dockerfile -t rag-api:latest .

# Tag for ECR
docker tag rag-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-api:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-api:latest

# Update ECS service
aws ecs update-service \
  --cluster rag-system-cluster \
  --service rag-api \
  --force-new-deployment
```

## 📊 Monitoring

### CloudWatch Dashboards

Access CloudWatch console to view:
- Request latency (p50, p90, p99)
- Error rates
- Token usage
- Cost metrics
- Evaluation scores

### LangSmith Tracing

Access LangSmith dashboard to:
- Trace individual requests
- Debug retrieval and generation
- Analyze prompt performance
- Compare model versions

### Logs

```bash
# Stream API logs
aws logs tail /ecs/rag-api --follow

# Query specific request
aws logs filter-pattern '{$.request_id = "req_abc123"}' /ecs/rag-api
```

## 🔄 Feedback Loop

### 1. Collect Feedback

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req_abc123",
    "rating": -1,
    "comment": "Answer was not helpful"
  }'
```

### 2. Export Negative Feedback to Eval Set

```bash
poetry run python scripts/export_feedback.py \
  --lookback-days 7 \
  --min-samples 10
```

This creates a JSONL file in S3 with unannotated examples.

### 3. Human Annotation

Review exported examples and add:
- `expected_answer`: What the correct answer should be
- `ground_truth_docs`: Which documents should have been retrieved

### 4. Add to Evaluation Dataset

Append annotated examples to `eval/datasets/eval_v1.jsonl`

### 5. Run Regression Tests

```bash
poetry run python scripts/run_evaluation.py \
  --dataset eval/datasets/eval_v1.jsonl
```

New examples are now regression tests. Future changes must maintain or improve these metrics.

## 🔧 Configuration

Key configuration in `.env`:

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db
DB_USER=rag_user
DB_PASSWORD=rag_password

# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-secret>

# Bedrock
BEDROCK_LLM_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0

# RAG
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=50
RAG_TOP_K=20
RAG_RERANK_TOP_K=5
RAG_VECTOR_WEIGHT=0.6
RAG_BM25_WEIGHT=0.4

# Observability
LANGSMITH_API_KEY=<your-key>
LOG_LEVEL=INFO
```

## 📈 Scalability

### Horizontal Scaling

- ECS Auto-scaling: Configure based on CPU, memory, or request count
- RDS Read Replicas: Add read replicas for read-heavy workloads

### Performance Optimization

- **Caching**: Cache embeddings and common queries in Redis
- **Batch Processing**: Batch embed operations (up to 128 texts)
- **Index Optimization**: Use HNSW index for faster vector search
- **Model Selection**: Route simple queries to Haiku, complex to Sonnet

### Multi-Tenancy

Add `tenant_id` to all tables and implement row-level security:

```sql
ALTER TABLE documents ADD COLUMN tenant_id TEXT NOT NULL;
CREATE INDEX idx_documents_tenant ON documents(tenant_id);
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
```

## 🔒 Security

- **IAM Roles**: Use task roles for ECS, no hardcoded credentials
- **Secrets Management**: Store secrets in AWS Secrets Manager
- **Network Isolation**: Private subnets for ECS and RDS
- **Encryption**: Enable encryption at rest for RDS and S3
- **API Authentication**: Implement API key or OAuth2

## 🐛 Troubleshooting

### Database Connection Issues

```bash
# Check if Postgres is running
docker-compose ps postgres

# Check connection from container
docker-compose exec api psql -h postgres -U rag_user -d rag_db

# View logs
docker-compose logs postgres
```

### Bedrock Access Issues

```bash
# Check AWS credentials
aws sts get-caller-identity

# Test Bedrock access
aws bedrock list-foundation-models --region us-east-1

# Verify model access
aws bedrock invoke-model \
  --model-id anthropic.claude-3-5-sonnet-20241022-v2:0 \
  --body '{"anthropic_version":"bedrock-2023-05-31","messages":[{"role":"user","content":"Hello"}]}' \
  --region us-east-1 \
  output.json
```

### High Latency

- Check CloudWatch metrics for bottlenecks
- Review LangSmith traces for slow nodes
- Consider adding caching layer
- Optimize retrieval (reduce top_k, improve indexing)

## 📚 Further Reading

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [DeepEval Documentation](https://docs.confident-ai.com/)
- [RAGAS Documentation](https://docs.ragas.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks: `poetry run pytest && poetry run ruff check .`
5. Run evaluation: `poetry run python scripts/run_evaluation.py`
6. Submit a pull request

## 📄 License

[Your License Here]

## 👥 Team

[Your Team Information]

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Contact: team@company.com
- Slack: #rag-system
