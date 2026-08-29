#!/usr/bin/env python3
"""
CLI launcher: Script for training the MILM model
Usage: python scripts/train.py [config_path]
"""

import sys
import os
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
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / "config.yaml")
    print(f"Loading configuration from: {config_path}")
    m_cfg, t_cfg = load_config(config_path)
    
    # Set working directory to project root
    os.chdir(ROOT_DIR)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    use_nvtx = t_cfg.emit_nvtx and torch.cuda.is_available()
    with torch.autograd.profiler.emit_nvtx(enabled=use_nvtx):
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            train(m_cfg, t_cfg)

if __name__ == "__main__":
    main()

