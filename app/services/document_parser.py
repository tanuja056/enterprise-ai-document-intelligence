from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument

def parse_document(path: str) -> list[dict]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    pages = []
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        for index, page in enumerate(reader.pages, start=1):
            pages.append({"text": page.extract_text() or "", "page": index})
    elif suffix == ".docx":
        doc = DocxDocument(str(file_path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        pages.append({"text": text, "page": None})
    elif suffix == ".txt":
        pages.append({"text": file_path.read_text(encoding="utf-8", errors="ignore"), "page": None})
    else:
        raise ValueError("Unsupported file type. Upload PDF, DOCX, or TXT.")
    return [p for p in pages if p["text"].strip()]

def clean_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())

def chunk_pages(pages: list[dict], chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    chunks = []
    for page_data in pages:
        text = clean_text(page_data["text"])
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"text": chunk, "page": page_data["page"]})
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
