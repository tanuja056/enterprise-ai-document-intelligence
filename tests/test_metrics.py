import pytest
from evaluation.metrics import recall_at_k, reciprocal_rank_at_k, mean_reciprocal_rank

def test_recall_at_k():
    assert recall_at_k(["A", "B", "C"], {"A", "C"}, 2) == 0.5

def test_reciprocal_rank_at_k():
    assert reciprocal_rank_at_k(["A", "B", "C"], {"B"}, 3) == 0.5

def test_mean_reciprocal_rank():
    results = [["A", "B"], ["C", "D"]]
    relevant = [{"B"}, {"C"}]
    assert mean_reciprocal_rank(results, relevant, 2) == 0.75

def test_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, 0)
