from pathlib import Path
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app.config import settings
from app.db import init_db, SessionLocal, Document
from app.schemas import QueryRequest, QueryResponse, Citation, IngestResponse
from app.services.document_parser import parse_document, chunk_pages
from app.services.retrieval import add_chunks, retrieve
from app.services.generation import generate_answer

app = FastAPI(
    title="Enterprise AI Document Intelligence",
    description="Grounded enterprise document question answering with hybrid RAG retrieval and citations.",
    version="1.0.0"
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup():
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
    init_db()

@app.get("/")
def root():
    return {"name": "Enterprise AI Document Intelligence", "status": "running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/documents/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(400, "Only PDF, DOCX, and TXT files are supported.")
    content = await file.read()
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_file_size_mb} MB limit.")
    safe_name = Path(file.filename or "upload").name
    destination = Path(settings.upload_dir) / safe_name
    # Avoid overwriting a previously uploaded file with the same name.
    if destination.exists():
        destination = destination.with_name(f"{destination.stem}_{abs(hash(content))}{destination.suffix}")
    destination.write_bytes(content)
    try:
        pages = parse_document(str(destination))
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("No extractable text found in document.")
        record = Document(
            filename=safe_name,
            stored_path=str(destination),
            content_type=file.content_type or "application/octet-stream",
            chunk_count=0,
            status="processing"
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        count = add_chunks(record.id, safe_name, chunks)
        record.chunk_count = count
        record.status = "processed"
        db.commit()
        return IngestResponse(document_id=record.id, filename=safe_name, chunks_created=count, status="processed")
    except Exception as exc:
        db.rollback()
        destination.unlink(missing_ok=True)
        raise HTTPException(422, f"Document processing failed: {exc}")

@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    records = db.query(Document).order_by(Document.created_at.desc()).all()
    return [{
        "id": r.id, "filename": r.filename, "chunk_count": r.chunk_count,
        "status": r.status, "created_at": r.created_at.isoformat()
    } for r in records]

@app.post("/query", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    results = retrieve(request.question, request.top_k)
    answer, model_name = generate_answer(request.question, results)
    citations = []
    for item in results:
        meta = item["metadata"]
        page = meta.get("page", -1)
        citations.append(Citation(
            document=meta["filename"],
            page=page if page > 0 else None,
            chunk_id=f"{meta['document_id']}:{meta['chunk_index']}",
            score=round(item.get("rerank_score", item.get("hybrid_score", 0.0)), 4),
            excerpt=item["text"][:350]
        ))
    # This is a retrieval-grounding indicator, not a guarantee of factual correctness.
    grounded = bool(results) and model_name != "no-context"
    return QueryResponse(
        answer=answer, citations=citations, retrieval_count=len(results),
        grounded=grounded, model=model_name
    )
