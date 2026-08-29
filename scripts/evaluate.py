#!/usr/bin/env python3
"""
CLI launcher: Quantitative and qualitative evaluation script for MILM
Usage: python scripts/evaluate.py
"""

import sys
import os
from pathlib import Path

# Add project root and src directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from src.config import load_config
from src.evaluate import LLMEvaluator

def main():
    ckpt_path = str(ROOT_DIR / "checkpoints" / "best_model.pt")
    tok_path = str(ROOT_DIR / "checkpoints" / "tokenizer.json")
    config_path = str(ROOT_DIR / "config.yaml")
    
    _, train_config = load_config(config_path)
    
    if not (os.path.exists(ckpt_path) and os.path.exists(tok_path)):
        print(f"Model checkpoint ({ckpt_path}) or tokenizer ({tok_path}) not found.")
        print("Please run 'python scripts/train.py' first.")
        sys.exit(1)

    evaluator = LLMEvaluator(checkpoint_path=ckpt_path, tokenizer_path=tok_path, device=train_config.device)
    
    print("\n==========================================")
    print(" 1. Quantitative Evaluation: Perplexity (PPL)")
    print("==========================================")
    sample_eval_corpus = (
        "The artificial intelligence and transformer architecture revolutionized natural language processing. "
        "Self-attention allows the model to weigh the importance of different tokens dynamically."
    ) * 10
    loss, ppl = evaluator.evaluate_perplexity(sample_eval_corpus)
    print(f"Test Loss: {loss:.4f} | Perplexity: {ppl:.2f}")

    print("\n==========================================")
    print(" 2. Similarity Evaluation (BLEU / ROUGE-L)")
    print("==========================================")
    test_pairs = [
        ("The artificial ", "intelligence and transformer architecture"),
        ("Self-attention allows ", "the model to weigh the importance"),
    ]
    sim_scores = evaluator.evaluate_similarity_metrics(test_pairs)
    print(f"BLEU-4 Score : {sim_scores['BLEU']:.4f}")
    print(f"ROUGE-L Score: {sim_scores['ROUGE-L']:.4f}")

    print("\n==========================================")
    print(" 3. Prompt Benchmark Testbed Results")
    print("==========================================")
    benchmark_prompts = [
        "The transformer ",
        "Residual connections "
    ]
    bench_results = evaluator.run_benchmark_suite(benchmark_prompts, temperatures=[0.2, 0.7, 1.2])
    for item in bench_results:
        print(f"\nPrompt: '{item['prompt']}'")
        for temp_k, gen_info in item['generations'].items():
            print(f"  [{temp_k}] (Repetition Rate: {gen_info['repetition_rate']:.2f}) -> {gen_info['text'].strip()}")

if __name__ == "__main__":
    main()
