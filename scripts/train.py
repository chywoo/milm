#!/usr/bin/env python3
"""
CLI 런처: MILM 모델 학습 실행 스크립트
사용법: python scripts/train.py [설정파일_경로]
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
from src.train import train

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / "config.yaml")
    print(f"설정 로딩 중: {config_path}")
    m_cfg, t_cfg = load_config(config_path)
    
    # 작업 디렉토리를 프로젝트 루트 기준으로 설정
    os.chdir(ROOT_DIR)
    train(m_cfg, t_cfg)

if __name__ == "__main__":
    main()
