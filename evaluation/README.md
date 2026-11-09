# Retrieval and answer evaluation

Use a manually curated set of questions with expected source documents and expected answer facts.

Recommended metrics:
- Recall@K: fraction of relevant chunks retrieved in the top K.
- Precision@K: fraction of retrieved chunks that are relevant.
- MRR: reciprocal rank of the first relevant result.
- Faithfulness: whether answer claims are supported by retrieved context.
- Answer relevance: whether the answer addresses the question.

Do not report evaluation scores until you have run a labeled evaluation set. Add a `golden_questions.jsonl` file with question, relevant_document, and expected_facts fields as the corpus grows.
