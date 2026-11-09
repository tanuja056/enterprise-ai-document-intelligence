import re
import uuid
from functools import lru_cache
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.config import settings

@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(settings.embedding_model)

@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(settings.reranker_model)

_client = None
_collection = None

def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_path)
        _collection = _client.get_or_create_collection(
            name="enterprise_documents",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def add_chunks(document_id: int, filename: str, chunks: list[dict]):
    if not chunks:
        return 0
    collection = get_collection()
    texts = [c["text"] for c in chunks]
    embeddings = get_embedder().encode(texts, normalize_embeddings=True).tolist()
    ids, metadatas = [], []
    for i, chunk in enumerate(chunks):
        ids.append(str(uuid.uuid4()))
        metadatas.append({
            "document_id": document_id,
            "filename": filename,
            "page": chunk["page"] if chunk["page"] is not None else -1,
            "chunk_index": i
        })
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return len(texts)

def retrieve(question: str, top_k: int | None = None):
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    k = min(max(top_k or settings.top_k, 30), count)
    query_embedding = get_embedder().encode([question], normalize_embeddings=True).tolist()
    semantic = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    docs = semantic["documents"][0]
    metas = semantic["metadatas"][0]
    distances = semantic["distances"][0]

    # Lexical BM25 over the indexed corpus; fuse with semantic candidates.
    all_rows = collection.get(include=["documents", "metadatas"])
    corpus = all_rows["documents"]
    all_metas = all_rows["metadatas"]
    tokens = lambda s: re.findall(r"\w+", s.lower())
    bm25 = BM25Okapi([tokens(d) for d in corpus]) if corpus else None
    lexical_scores = bm25.get_scores(tokens(question)) if bm25 else []
    lexical_rank = sorted(range(len(lexical_scores)), key=lambda i: lexical_scores[i], reverse=True)[:k]

    candidates = {}
    for text, meta, distance in zip(docs, metas, distances):
        key = (meta["document_id"], meta["chunk_index"])
        candidates[key] = {
            "text": text, "metadata": meta,
            "semantic_score": 1.0 - float(distance),
            "hybrid_score": 1.0 - float(distance)
        }

    # Add top lexical matches not already selected.
    for idx in lexical_rank:
        meta, text = all_metas[idx], corpus[idx]
        key = (meta["document_id"], meta["chunk_index"])
        lexical_bonus = 0.15
        if key in candidates:
            candidates[key]["hybrid_score"] += lexical_bonus
        else:
            candidates[key] = {
                "text": text, "metadata": meta,
                "semantic_score": 0.0, "hybrid_score": lexical_bonus
            }

    ranked = list(candidates.values())
    if not ranked:
        return []

    pairs = [(question, item["text"]) for item in ranked]
    scores = get_reranker().predict(pairs)

    for item, score in zip(ranked, scores):
        item["rerank_score"] = float(score)

    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    final_k = min(top_k or settings.rerank_top_k, len(ranked))
    return ranked[:final_k]
