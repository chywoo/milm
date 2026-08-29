import os
import tempfile
import pytest
from src.config import ModelConfig, TrainConfig, load_config, save_config, get_default_device

def test_default_configs():
    m_cfg = ModelConfig()
    t_cfg = TrainConfig()
    
    assert m_cfg.d_model == 256
    assert m_cfg.num_heads == 8
    assert m_cfg.seq_len == 128
    assert m_cfg.num_layers == 6
    assert t_cfg.batch_size == 64
    assert t_cfg.epochs == 10
    assert t_cfg.device in ("cuda", "mps", "cpu")

def test_model_config_from_dict():
    data = {
        "vocab_size": "512",
        "seq_len": "256",
        "d_model": "128",
        "num_heads": "4",
        "num_layers": "2",
        "d_ff": "512",
        "dropout": "0.2",
        "emit_nvtx": True,
        "unknown_key": "ignore_me"
    }
    cfg = ModelConfig.from_dict(data)
    assert cfg.vocab_size == 512
    assert cfg.seq_len == 256
    assert cfg.d_model == 128
    assert cfg.num_heads == 4
    assert cfg.num_layers == 2
    assert cfg.d_ff == 512
    assert cfg.dropout == 0.2
    assert cfg.emit_nvtx is True

def test_train_config_from_dict():
    data = {
        "batch_size": "32",
        "epochs": "5",
        "learning_rate": "1e-3",
        "device": "cpu",
        "use_amp": False,
        "emit_nvtx": True
    }
    cfg = TrainConfig.from_dict(data)
    assert cfg.batch_size == 32
    assert cfg.epochs == 5
    assert cfg.learning_rate == 0.001
    assert cfg.device == "cpu"
    assert cfg.use_amp is False
    assert cfg.emit_nvtx is True

def test_config_save_and_load():
    m_cfg = ModelConfig(vocab_size=100, seq_len=64, d_model=128)
    t_cfg = TrainConfig(batch_size=16, epochs=3, device="cpu")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_config.yaml")
        save_config(m_cfg, t_cfg, config_path)
        
        loaded_m, loaded_t = load_config(config_path)
        assert loaded_m.vocab_size == 100
        assert loaded_m.seq_len == 64
        assert loaded_m.d_model == 128
        assert loaded_t.batch_size == 16
        assert loaded_t.epochs == 3
        assert loaded_t.device == "cpu"

def test_load_config_fallback_nonexistent():
    m_cfg, t_cfg = load_config("nonexistent_path_xyz.yaml")
    assert isinstance(m_cfg, ModelConfig)
    assert isinstance(t_cfg, TrainConfig)

