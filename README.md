# Enterprise AI Document Intelligence Platform

An end-to-end Retrieval-Augmented Generation (RAG) application for searching enterprise PDF, DOCX, and TXT documents and answering questions with source-grounded citations.

## Features

- PDF, DOCX, and TXT ingestion
- Text extraction, normalization, overlapping chunking
- Local Sentence Transformers embeddings
- Persistent ChromaDB vector storage
- Hybrid retrieval: dense semantic search + BM25 lexical retrieval
- Cross-encoder reranking
- OpenAI-powered context-constrained question answering
- Source citations with document names and page references
- FastAPI REST API and interactive Swagger docs
- PostgreSQL metadata persistence (SQLite fallback for local development)
- Docker Compose deployment
- Basic API tests

## Architecture

See [`docs/architecture.md`](docs/architecture.md).

## Quick start (local)

Requires Python 3.11+.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/enterprise-ai-document-intelligence.git
cd enterprise-ai-document-intelligence

python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env
```

For local development, set `DATABASE_URL=sqlite:///./document_intelligence.db` in `.env`.
Set `OPENAI_API_KEY` to enable generated answers. Without it, the API returns retrieved evidence instead of inventing an LLM answer.

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs.

## API workflow

1. `POST /documents/upload` — upload a PDF, DOCX, or TXT.
2. `GET /documents` — list indexed documents.
3. `POST /query` — ask a question.

Example query body:

```json
{
  "question": "What is the company's parental leave policy?",
  "top_k": 8
}
```

## Docker

Copy `.env.example` to `.env`. The Compose configuration starts PostgreSQL and the API.

```bash
docker compose up --build
```

The first startup downloads embedding and reranker models; allow extra time and memory.

## Run tests

```bash
pytest -q
```

## Project status

The initial end-to-end API implementation is provided. Validate it in your environment, add representative test documents and a labeled evaluation set, and verify cloud deployment before describing those capabilities as production-deployed.

## Security and limitations

- Do not upload confidential or personally identifiable documents to a public demo.
- Configure secrets through environment variables; never commit `.env`.
- This starter uses a local persistent Chroma collection. Multi-tenant access controls, authentication, deletion workflows, OCR for scanned PDFs, and production-grade observability are future hardening tasks.
- `grounded` means retrieval returned context; it does not certify that every generated claim is correct.
