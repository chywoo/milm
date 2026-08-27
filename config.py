from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ModelConfig:
    """트랜스포머 모델 아키텍처 하이퍼파라미터"""
    vocab_size: int = 256         # 문자 집합 또는 토큰 수 (동적 설정 가능)
    seq_len: int = 128            # 최대 문맥 길이 (Context Window)
    d_model: int = 256            # 임베딩 및 히든 차원
    num_heads: int = 8            # Multi-Head 수
    num_layers: int = 6           # 트랜스포머 블록 레이어 수
    d_ff: int = 1024              # FFN 내부 확장 차원 (일반적으로 4 * d_model). 논문에서는 2048
    dropout: float = 0.1     # 드롭아웃 비율

@dataclass
class TrainConfig:
    """학습 및 인프라 최적화 설정"""
    data_dir: str = "data"        # 훈련 텍스트 파일들이 위치한 디렉토리
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best_model.pt"
    
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0        # 기울기 폭주 방지 클리핑
    
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    use_amp: bool = True          # Automatic Mixed Precision 활성화
    compile_model: bool = True    # PyTorch 2.0+ torch.compile 활성화 (CUDA 전용)
    val_split: float = 0.1   # 검증 데이터셋 비율