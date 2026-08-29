#!/usr/bin/env python3
"""
CLI launcher: Script for training the MILM model
Usage: python scripts/train.py [config_path]
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

from src.config import load_config
from src.train import train

def main():
    parser = argparse.ArgumentParser(description="MILM Model Training CLI")
    parser.add_argument("config", nargs="?", default=str(ROOT_DIR / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--profile", action="store_true", default=False, help="Enable PyTorch Profiler")
    parser.add_argument("--profile_dir", type=str, default="profiler_logs", help="Directory for profiling trace output")
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    m_cfg, t_cfg = load_config(args.config)
    
    # Set working directory to project root
    os.chdir(ROOT_DIR)

    enable_profile = args.profile or t_cfg.profile
    profile_dir = args.profile_dir if args.profile else t_cfg.profile_dir

    if enable_profile:
        print(f"PyTorch Profiler enabled. Traces will be saved to: {profile_dir}")
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available() and t_cfg.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            train(m_cfg, t_cfg, on_step_end=lambda step: prof.step())
    else:
        train(m_cfg, t_cfg)

if __name__ == "__main__":
    main()

