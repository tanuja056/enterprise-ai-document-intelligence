def recall_at_k(ranked_results: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant documents retrieved in the top k results."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not relevant_ids:
        return 0.0

    retrieved = set(ranked_results[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def reciprocal_rank_at_k(
    ranked_results: list[str], relevant_ids: set[str], k: int
) -> float:
    """Reciprocal rank of the first relevant result within the top k."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    for rank, result_id in enumerate(ranked_results[:k], start=1):
        if result_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    query_results: list[list[str]],
    relevant_by_query: list[set[str]],
    k: int
) -> float:
    """Average reciprocal rank across multiple queries."""
    if len(query_results) != len(relevant_by_query):
        raise ValueError("Each query must have a corresponding relevance set")
    if not query_results:
        return 0.0

    scores = [
        reciprocal_rank_at_k(results, relevant, k)
        for results, relevant in zip(query_results, relevant_by_query)
    ]
    return sum(scores) / len(scores)
