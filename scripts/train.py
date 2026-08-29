#!/usr/bin/env python3
"""
CLI launcher: Script for training the MILM model
Usage: python scripts/train.py [config_path]
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
from src.train import train

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / "config.yaml")
    print(f"Loading configuration from: {config_path}")
    m_cfg, t_cfg = load_config(config_path)
    
    # Set working directory to project root
    os.chdir(ROOT_DIR)
    train(m_cfg, t_cfg)

if __name__ == "__main__":
    main()
