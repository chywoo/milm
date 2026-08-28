#!/usr/bin/env python3
"""
CLI 런처: MILM 모델 성능 정량/정성 평가 스크립트
사용법: python scripts/evaluate.py
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 및 src 디렉토리를 sys.path에 추가
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
        print(f"모델 체크포인트({ckpt_path}) 또는 토크나이저({tok_path})가 존재하지 않습니다.")
        print("먼저 'python scripts/train.py'를 실행하세요.")
        sys.exit(1)

    evaluator = LLMEvaluator(checkpoint_path=ckpt_path, tokenizer_path=tok_path, device=train_config.device)
    
    print("\n==========================================")
    print(" 1. Perplexity (PPL) 정량 평가")
    print("==========================================")
    sample_eval_corpus = (
        "The artificial intelligence and transformer architecture revolutionized natural language processing. "
        "Self-attention allows the model to weigh the importance of different tokens dynamically."
    ) * 10
    loss, ppl = evaluator.evaluate_perplexity(sample_eval_corpus)
    print(f"Test Loss: {loss:.4f} | Perplexity: {ppl:.2f}")

    print("\n==========================================")
    print(" 2. 문장 유사도 평가 (BLEU / ROUGE-L)")
    print("==========================================")
    test_pairs = [
        ("The artificial ", "intelligence and transformer architecture"),
        ("Self-attention allows ", "the model to weigh the importance"),
    ]
    sim_scores = evaluator.evaluate_similarity_metrics(test_pairs)
    print(f"BLEU-4 Score : {sim_scores['BLEU']:.4f}")
    print(f"ROUGE-L Score: {sim_scores['ROUGE-L']:.4f}")

    print("\n==========================================")
    print(" 3. 프롬프트 벤치마크 테스트베드 결과")
    print("==========================================")
    benchmark_prompts = [
        "The transformer ",
        "Residual connections "
    ]
    bench_results = evaluator.run_benchmark_suite(benchmark_prompts, temperatures=[0.2, 0.7, 1.2])
    for item in bench_results:
        print(f"\n프롬프트: '{item['prompt']}'")
        for temp_k, gen_info in item['generations'].items():
            print(f"  [{temp_k}] (반복률: {gen_info['repetition_rate']:.2f}) -> {gen_info['text'].strip()}")

if __name__ == "__main__":
    main()
