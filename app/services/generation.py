from openai import OpenAI
from app.config import settings

def generate_answer(question: str, contexts: list[dict]):
    if not contexts:
        return ("I could not find relevant information in the uploaded documents.", "no-context")
    context_blocks = []
    for i, item in enumerate(contexts, start=1):
        meta = item["metadata"]
        page = meta.get("page", -1)
        page_label = f"page {page}" if page and page > 0 else "page not available"
        context_blocks.append(
            f"[Source {i}: {meta['filename']}, {page_label}]\n{item['text']}"
        )
    context = "\n\n".join(context_blocks)

    if not settings.openai_api_key:
        # Safe offline fallback: return retrieved evidence rather than fabricate an answer.
        excerpts = "\n\n".join(f"- {b}" for b in context_blocks[:3])
        return ("OPENAI_API_KEY is not configured. Retrieved evidence:\n\n" + excerpts, "retrieval-only")

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        messages=[
            {"role": "system", "content":
             "Answer only using the supplied document context. If the answer is absent, say you cannot find it. "
             "Do not use outside knowledge. Cite supporting sources inline as [Source 1], [Source 2]."},
            {"role": "user", "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}"}
        ]
    )
    return response.choices[0].message.content or "", settings.openai_model
