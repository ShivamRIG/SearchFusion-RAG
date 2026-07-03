# Hybrid RAG System

A **production-ready Hybrid Retrieval-Augmented Generation (RAG)** application built using **LangChain**, **Qdrant**, **Elasticsearch**, **FastAPI**, **Langfuse**, **Redis**, and **OpenAI**.

The system combines **semantic search (dense retrieval)** with **keyword search (BM25)**, applies **Reciprocal Rank Fusion (RRF)** and **BGE reranking**, generates answers using an LLM, and provides **observability**, **streaming**, **caching**, and **evaluation**.

---

# Features

- Hybrid Retrieval (Dense + BM25)
- Qdrant Vector Database
- Elasticsearch BM25 Search
- LangChain Integration
- OpenAI GPT Models
- OpenAI Embeddings
- BGE Cross Encoder Reranker
- Langfuse Observability
- Redis Response Caching
- Streaming Responses
- FastAPI REST API
- Document Ingestion Pipeline
- Prompt Engineering
- Evaluation Module
- Modular Project Structure
- Docker Compose Support

---

# Architecture

```text
                     User
                       │
                       ▼
                 FastAPI API
                       │
                       ▼
               Hybrid RAG Pipeline
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Redis Cache       Langfuse Trace
              │
              ▼
       Query Embedding
              │
              ▼
      Hybrid Retrieval
      ┌────────┴─────────┐
      ▼                  ▼
 Qdrant (Dense)   Elasticsearch (BM25)
      │                  │
      └────────┬─────────┘
               ▼
     Reciprocal Rank Fusion
               ▼
        BGE Reranker
               ▼
       Prompt Construction
               ▼
         OpenAI GPT Model
               ▼
         Generated Answer
               ▼
     Evaluation + Langfuse
               ▼
          Cached Response
               ▼
             Client
```

---

# Project Structure

```text
hybrid-rag-system/
│
├── app/
│   ├── cache.py
│   ├── config.py
│   ├── evaluation.py
│   ├── generation.py
│   ├── ingestion.py
│   ├── langfuse_client.py
│   ├── main.py
│   ├── pipeline.py
│   ├── reranker.py
│   └── retrieval.py
│
├── data/
│   └── documents/
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```

---

# Tech Stack

| Technology | Purpose |
|------------|---------|
| FastAPI | REST API |
| LangChain | LLM Orchestration |
| OpenAI GPT | Text Generation |
| OpenAI Embeddings | Dense Embeddings |
| Qdrant | Vector Database |
| Elasticsearch | BM25 Search |
| Redis | Response Cache |
| Langfuse | Observability |
| Docker | Infrastructure |
| Sentence Transformers | BGE Reranker |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/<your-username>/hybrid-rag-system.git

cd hybrid-rag-system
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=documents

ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=documents

EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

---

# Run Docker Services

```bash
docker compose up -d
```

This starts:

- Qdrant
- Elasticsearch
- Redis
- Langfuse
- PostgreSQL

---

# Add Documents

```text
data/
└── documents/
    ├── handbook.pdf
    ├── policy.pdf
    ├── faq.txt
    └── notes.md
```

---

# Ingest Documents

```bash
python -m app.ingestion
```

Example output:

```text
Loaded 5 documents

Created 84 chunks

Indexed into Qdrant

Indexed into Elasticsearch

Ingestion Complete
```

---

# Run FastAPI

```bash
uvicorn app.main:app --reload
```

Server:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

# API Endpoints

## Health

```
GET /health
```

Response:

```json
{
  "status": "healthy"
}
```

---

## Chat

```
POST /chat
```

Request:

```json
{
  "question": "What is the refund policy?"
}
```

Response:

```json
{
  "answer": "The refund policy allows customers..."
}
```

---

## Streaming Chat

```
POST /chat/stream
```

Returns a streaming text response.

---

# Pipeline Workflow

```text
User Question
      │
      ▼
Redis Cache
      │
      ▼
Query Embedding
      │
      ▼
Dense Search (Qdrant)
      │
      ▼
Sparse Search (BM25)
      │
      ▼
Reciprocal Rank Fusion
      │
      ▼
BGE Reranker
      │
      ▼
Prompt Construction
      │
      ▼
OpenAI GPT
      │
      ▼
Evaluation
      │
      ▼
Langfuse Logging
      │
      ▼
Return Response
```

---

# Hybrid Retrieval

The retrieval pipeline combines:

### Dense Retrieval

- OpenAI Embeddings
- Qdrant
- Semantic Similarity Search

### Sparse Retrieval

- Elasticsearch
- BM25 Keyword Search

### Fusion

Reciprocal Rank Fusion (RRF)

```text
Dense Results

A
B
C

BM25 Results

B
D
A

↓

Final Ranking

B
A
C
D
```

---

# Reranking

The retrieved documents are reranked using:

```text
BAAI/bge-reranker-base
```

Benefits:

- Better Context Selection
- Higher Answer Quality
- Improved LLM Accuracy

---

# Redis Cache

Frequently asked questions are cached.

```text
Question

↓

Redis

↓

Cached Answer

↓

Return
```

Benefits:

- Faster Responses
- Reduced API Cost
- Lower Latency

---

# Langfuse Observability

The project logs:

- User Question
- Retrieved Documents
- Final Prompt
- Generated Answer
- Metadata
- Evaluation Scores

Example metadata:

```json
{
  "retriever": "hybrid",
  "reranker": "bge",
  "documents": 5
}
```

---

# Evaluation

The evaluation module measures:

- Faithfulness
- Relevance

Results are automatically logged into Langfuse.

---

# Future Improvements

- Multi-Query Retrieval
- Query Expansion
- Hybrid Search Weights
- Metadata Filtering
- Authentication
- Conversation Memory
- Citation Generation
- RAGAS Evaluation
- Kubernetes Deployment
- CI/CD Pipeline
- AWS Deployment

---

# Requirements

- Python 3.11+
- Docker
- OpenAI API Key

---

# Screenshots

You can include screenshots of:

- Swagger UI
- Langfuse Dashboard
- Qdrant Dashboard
- Elasticsearch
- API Responses

inside a `screenshots/` folder.

---

# Contributing

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/new-feature
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push your branch.

```bash
git push origin feature/new-feature
```

5. Open a Pull Request.

---

# License

This project is licensed under the MIT License.

---

# Author

**Your Name**

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile

---

If you found this project useful, consider giving it a star.