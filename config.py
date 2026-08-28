import os
from dataclasses import dataclass, asdict, fields
from typing import Tuple, Dict, Any, Optional
import yaml
import torch

@dataclass
class ModelConfig:
    """트랜스포머 모델 아키텍처 하이퍼파라미터"""
    vocab_size: int = 256         # 문자 집합 또는 토큰 수 (동적 설정 가능)
    seq_len: int = 128            # 최대 문맥 길이 (Context Window)
    d_model: int = 256            # 임베딩 및 히든 차원
    num_heads: int = 8            # Multi-Head 수
    num_layers: int = 6           # 트랜스포머 블록 레이어 수
    d_ff: int = 1024              # FFN 내부 확장 차원 (일반적으로 4 * d_model)
    dropout: float = 0.1          # 드롭아웃 비율

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """딕셔너리로부터 유효한 필드만 추출하여 ModelConfig 인스턴스 생성"""
        if not data:
            return cls()
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys and v is not None}
        
        # 타입 캐스팅
        if "vocab_size" in filtered:
            filtered["vocab_size"] = int(filtered["vocab_size"])
        if "seq_len" in filtered:
            filtered["seq_len"] = int(filtered["seq_len"])
        if "d_model" in filtered:
            filtered["d_model"] = int(filtered["d_model"])
        if "num_heads" in filtered:
            filtered["num_heads"] = int(filtered["num_heads"])
        if "num_layers" in filtered:
            filtered["num_layers"] = int(filtered["num_layers"])
        if "d_ff" in filtered:
            filtered["d_ff"] = int(filtered["d_ff"])
        if "dropout" in filtered:
            filtered["dropout"] = float(filtered["dropout"])

        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """ModelConfig 객체를 딕셔너리로 변환"""
        return asdict(self)


def get_default_device() -> str:
    """사용 가능한 최적의 하드웨어 가속 디바이스 자동 감지"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    """학습 및 인프라 최적화 설정"""
    data_dir: str = "data"        # 훈련 텍스트 파일들이 위치한 디렉토리
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best_model.pt"
    
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0        # 기울기 폭주 방지 클리핑
    
    device: str = "auto"          # "auto", "cuda", "mps", "cpu"
    use_amp: bool = True          # Automatic Mixed Precision 활성화
    compile_model: bool = True    # PyTorch 2.0+ torch.compile 활성화 (CUDA 전용)
    val_split: float = 0.1        # 검증 데이터셋 비율

    def __post_init__(self):
        if self.device == "auto" or not self.device:
            self.device = get_default_device()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        """딕셔너리로부터 유효한 필드만 추출하여 TrainConfig 인스턴스 생성"""
        if not data:
            return cls()
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys and v is not None}

        # 타입 캐스팅
        if "batch_size" in filtered:
            filtered["batch_size"] = int(filtered["batch_size"])
        if "epochs" in filtered:
            filtered["epochs"] = int(filtered["epochs"])
        if "learning_rate" in filtered:
            filtered["learning_rate"] = float(filtered["learning_rate"])
        if "min_lr" in filtered:
            filtered["min_lr"] = float(filtered["min_lr"])
        if "warmup_steps" in filtered:
            filtered["warmup_steps"] = int(filtered["warmup_steps"])
        if "weight_decay" in filtered:
            filtered["weight_decay"] = float(filtered["weight_decay"])
        if "grad_clip" in filtered:
            filtered["grad_clip"] = float(filtered["grad_clip"])
        if "use_amp" in filtered:
            filtered["use_amp"] = bool(filtered["use_amp"])
        if "compile_model" in filtered:
            filtered["compile_model"] = bool(filtered["compile_model"])
        if "val_split" in filtered:
            filtered["val_split"] = float(filtered["val_split"])
        if "device" in filtered and filtered["device"] == "auto":
            filtered["device"] = get_default_device()

        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """TrainConfig 객체를 딕셔너리로 변환"""
        return asdict(self)


def load_config(config_path: str = "config.yaml") -> Tuple[ModelConfig, TrainConfig]:
    """
    YAML 파일로부터 ModelConfig 및 TrainConfig를 로드합니다.
    config.yaml 파일이 존재하지 않는 경우 기본 설정을 반환합니다.
    """
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        model_data = data.get("model", {})
        train_data = data.get("train", {})
        
        m_cfg = ModelConfig.from_dict(model_data)
        t_cfg = TrainConfig.from_dict(train_data)
        return m_cfg, t_cfg
    
    # 설정 파일이 없을 경우 기본값 생성
    return ModelConfig(), TrainConfig()


def save_config(m_cfg: ModelConfig, t_cfg: TrainConfig, config_path: str = "config.yaml"):
    """ModelConfig 및 TrainConfig를 YAML 파일로 저장합니다."""
    data = {
        "model": m_cfg.to_dict(),
        "train": t_cfg.to_dict()
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
