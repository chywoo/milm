#!/usr/bin/env python3
"""
CLI launcher: Interactive text generation inference script for MILM
Usage: python scripts/infer.py [--prompt "Prompt text"] [--temp 0.7] [--tokens 200]
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import torch.profiler

# Add project root and src directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from src.infer import LLMInferenceEngine

def main():
    parser = argparse.ArgumentParser(description="MILM Text Generation Inference CLI")
    parser.add_argument("--prompt", type=str, default="The artificial ", help="Initial prompt text")
    parser.add_argument("--tokens", type=int, default=200, help="Maximum number of new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k filtering threshold")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) cumulative probability threshold")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT_DIR / "checkpoints" / "best_model.pt"), help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=str, default=str(ROOT_DIR / "checkpoints" / "tokenizer.json"), help="Tokenizer file path")
    parser.add_argument("--emit_nvtx", action="store_true", default=True, help="Enable NVTX marker emission for profiling")
    args = parser.parse_args()

    if not (os.path.exists(args.checkpoint) and os.path.exists(args.tokenizer)):
        print(f"Checkpoint file not found: {args.checkpoint}")
        print("Please train the model first by running 'python scripts/train.py'.")
        sys.exit(1)

    print(f"Loading checkpoint from: {args.checkpoint}")
    engine = LLMInferenceEngine(checkpoint_path=args.checkpoint, tokenizer_path=args.tokenizer)
    
    print(f"Prompt: \"{args.prompt}\"")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    use_nvtx = args.emit_nvtx and torch.cuda.is_available()
    with torch.autograd.profiler.emit_nvtx(enabled=use_nvtx):
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            output = engine.generate(
                prompt=args.prompt,
                max_new_tokens=args.tokens,
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )
    print("\n--- Generation Result ---")
    print(output)

if __name__ == "__main__":
    main()

