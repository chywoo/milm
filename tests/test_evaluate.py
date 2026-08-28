import pytest
from src.evaluate import calculate_ngram_precision, compute_bleu, compute_rouge_l, LLMEvaluator

def test_calculate_ngram_precision():
    ref = list("hello")
    cand = list("hello")
    assert calculate_ngram_precision(ref, cand, 1) == 1.0
    assert calculate_ngram_precision(ref, cand, 2) == 1.0

def test_compute_bleu():
    ref = "transformer architecture"
    cand_exact = "transformer architecture"
    assert pytest.approx(compute_bleu(ref, cand_exact), 0.001) == 1.0

    cand_empty = ""
    assert compute_bleu(ref, cand_empty) == 0.0

    cand_diff = "xyz 123"
    assert compute_bleu(ref, cand_diff) == 0.0

def test_compute_rouge_l():
    ref = "natural language processing"
    cand_exact = "natural language processing"
    assert compute_rouge_l(ref, cand_exact) == 1.0

    cand_empty = ""
    assert compute_rouge_l(ref, cand_empty) == 0.0
    assert compute_rouge_l("", "hello") == 0.0

    ref = "ABCDEFG"
    cand = "ABXDFG"
    score = compute_rouge_l(ref, cand)
    assert 0.0 < score < 1.0

