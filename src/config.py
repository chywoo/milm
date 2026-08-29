import os
from dataclasses import dataclass, asdict, fields
from typing import Tuple, Dict, Any, Optional
import yaml
import torch

@dataclass
class ModelConfig:
    """Transformer model architecture hyperparameters."""
    vocab_size: int = 256         # Vocabulary size / character count (dynamically updated)
    seq_len: int = 128            # Maximum context length (Context Window)
    d_model: int = 256            # Embedding and hidden dimension
    num_heads: int = 8            # Number of attention heads
    num_layers: int = 6           # Number of Transformer block layers
    d_ff: int = 1024              # FFN inner expansion dimension (typically 4 * d_model)
    dropout: float = 0.1          # Dropout rate
    emit_nvtx: bool = False       # Enable NVTX marker emission for profiler

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create a ModelConfig instance from a dictionary filtering valid fields."""
        if not data:
            return cls()
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys and v is not None}
        
        # Type casting
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
        if "emit_nvtx" in filtered:
            filtered["emit_nvtx"] = bool(filtered["emit_nvtx"])

        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelConfig instance to dictionary."""
        return asdict(self)


def get_default_device() -> str:
    """Auto-detect optimal hardware acceleration device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    """Training and infrastructure optimization configuration."""
    data_dir: str = "data"        # Directory containing training text files
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best_model.pt"
    
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0        # Gradient clipping max norm
    
    device: str = "auto"          # "auto", "cuda", "mps", "cpu"
    use_amp: bool = True          # Enable Automatic Mixed Precision
    compile_model: bool = True    # Enable PyTorch 2.0+ torch.compile (CUDA only)
    val_split: float = 0.1        # Validation dataset split ratio
    profile: bool = False         # Enable PyTorch Profiler
    profile_dir: str = "profiler_logs" # Directory for profiling traces
    emit_nvtx: bool = False       # Enable NVTX marker emission for profiler

    def __post_init__(self):
        if self.device == "auto" or not self.device:
            self.device = get_default_device()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        """Create a TrainConfig instance from a dictionary filtering valid fields."""
        if not data:
            return cls()
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys and v is not None}

        # Type casting
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
        if "profile" in filtered:
            filtered["profile"] = bool(filtered["profile"])
        if "profile_dir" in filtered:
            filtered["profile_dir"] = str(filtered["profile_dir"])
        if "emit_nvtx" in filtered:
            filtered["emit_nvtx"] = bool(filtered["emit_nvtx"])
        if "device" in filtered and filtered["device"] == "auto":
            filtered["device"] = get_default_device()

        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert TrainConfig instance to dictionary."""
        return asdict(self)


def load_config(config_path: str = "config.yaml") -> Tuple[ModelConfig, TrainConfig]:
    """
    Load ModelConfig and TrainConfig from a YAML file.
    Returns default configurations if the file does not exist.
    """
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        model_data = data.get("model", {})
        train_data = data.get("train", {})
        
        m_cfg = ModelConfig.from_dict(model_data)
        t_cfg = TrainConfig.from_dict(train_data)
        return m_cfg, t_cfg
    
    # Return defaults if configuration file does not exist
    return ModelConfig(), TrainConfig()


def save_config(m_cfg: ModelConfig, t_cfg: TrainConfig, config_path: str = "config.yaml"):
    """Save ModelConfig and TrainConfig to a YAML file."""
    data = {
        "model": m_cfg.to_dict(),
        "train": t_cfg.to_dict()
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

