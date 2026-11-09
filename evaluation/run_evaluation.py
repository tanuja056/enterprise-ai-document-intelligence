import json
from pathlib import Path
from app.services.retrieval import retrieve
from evaluation.metrics import recall_at_k, reciprocal_rank_at_k

dataset = json.loads(Path("evaluation/queries.json").read_text())
recalls, mrrs = [], []

for item in dataset:
    results = retrieve(item["question"], top_k=15)
    ranked = [
        f'{r["metadata"]["document_id"]}:{r["metadata"]["chunk_index"]}'
        for r in results
    ]
    relevant = set(item["relevant_chunks"])

    recall = recall_at_k(ranked, relevant, 3)
    mrr = reciprocal_rank_at_k(ranked, relevant, 3)
    recalls.append(recall)
    mrrs.append(mrr)

    print(f'\nQuestion: {item["question"]}')
    print(f'Recall@3: {recall:.3f} | MRR@3: {mrr:.3f}')

print("\n===== TOP-3 EVALUATION =====")
print(f'Average Recall@3: {sum(recalls)/len(recalls):.3f}')
print(f'Mean Reciprocal Rank@3: {sum(mrrs)/len(mrrs):.3f}')
