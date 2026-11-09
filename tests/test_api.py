from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upload_rejects_unsupported_file():
    response = client.post(
        "/documents/upload",
        files={"file": ("malware.exe", b"not an allowed document", "application/octet-stream")}
    )
    assert response.status_code == 400
def test_query_returns_answer_and_citations(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "retrieve", lambda question, top_k: [{
        "text": "The research uses machine learning classifiers.",
        "metadata": {
            "filename": "research.pdf",
            "page": 2,
            "document_id": 1,
            "chunk_index": 0
        },
        "rerank_score": 0.95
    }])

    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda question, results: (
            "The research uses machine learning classifiers.",
            "test-model"
        )
    )

    response = client.post(
        "/query",
        json={"question": "Which machine learning methods are used?", "top_k": 5}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    assert data["retrieval_count"] == 1
    assert data["citations"][0]["document"] == "research.pdf"
    assert data["citations"][0]["page"] == 2


def test_query_without_context(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "retrieve", lambda question, top_k: [])
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda question, results: ("No relevant context found.", "no-context")
    )

    response = client.post(
        "/query",
        json={"question": "What does this document explain?"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert data["retrieval_count"] == 0
    assert data["citations"] == []
