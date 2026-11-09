# Architecture

1. FastAPI accepts PDF, DOCX, and TXT uploads.
2. Parser extracts text; cleaner normalizes whitespace; chunker creates overlapping chunks.
3. Sentence Transformers creates normalized dense embeddings.
4. ChromaDB persists vectors and source metadata.
5. Query-time retrieval combines dense similarity candidates with BM25 lexical candidates.
6. A cross-encoder reranks candidates.
7. OpenAI generates a context-constrained response and cites source labels.
8. PostgreSQL stores document metadata; SQLite is available for local development.

The `grounded` response field indicates that supporting retrieval context was found. It is not a formal hallucination score.
