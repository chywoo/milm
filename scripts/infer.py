#!/usr/bin/env python3
"""
CLI 런처: MILM 대화형 텍스트 생성 추론 스크립트
사용법: python scripts/infer.py [--prompt "프롬프트"] [--temp 0.7] [--tokens 200]
"""

import sys
import os
import argparse
from pathlib import Path

# 프로젝트 루트 및 src 디렉토리를 sys.path에 추가
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from src.infer import LLMInferenceEngine

def main():
    parser = argparse.ArgumentParser(description="MILM 텍스트 생성 추론 CLI")
    parser.add_argument("--prompt", type=str, default="The artificial ", help="시작 프롬프트 텍스트")
    parser.add_argument("--tokens", type=int, default=200, help="최대 생성 토큰 수")
    parser.add_argument("--temp", type=float, default=0.7, help="샘플링 온도 (Temperature)")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k 필터링 개수")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (Nucleus) 누적 확률")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="반복 페널티")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT_DIR / "checkpoints" / "best_model.pt"), help="가중치 경로")
    parser.add_argument("--tokenizer", type=str, default=str(ROOT_DIR / "checkpoints" / "tokenizer.json"), help="토크나이저 경로")
    args = parser.parse_args()

    if not (os.path.exists(args.checkpoint) and os.path.exists(args.tokenizer)):
        print(f"체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")
        print("먼저 'python scripts/train.py'를 실행하여 모델을 학습하세요.")
        sys.exit(1)

    print(f"체크포인트 로드 중: {args.checkpoint}")
    engine = LLMInferenceEngine(checkpoint_path=args.checkpoint, tokenizer_path=args.tokenizer)
    
    print(f"프롬프트: \"{args.prompt}\"")
    output = engine.generate(
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )
    print("\n--- 생성 결과 ---")
    print(output)

if __name__ == "__main__":
    main()
