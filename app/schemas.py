from pydantic import BaseModel, Field
from typing import Any

class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)

class Citation(BaseModel):
    document: str
    page: int | None = None
    chunk_id: str
    score: float | None = None
    excerpt: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieval_count: int
    grounded: bool
    model: str

class IngestResponse(BaseModel):
    document_id: int
    filename: str
    chunks_created: int
    status: str

class DocumentResponse(BaseModel):
    id: int
    filename: str
    chunk_count: int
    status: str
    created_at: str
