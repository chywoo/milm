import pytest
import torch
from src.config import ModelConfig
from src.model import MiniLLM, TransformerBlock, CausalSelfAttention, FeedForward

@pytest.fixture
def small_config():
    return ModelConfig(
        vocab_size=50,
        seq_len=32,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        dropout=0.0
    )

def test_causal_self_attention(small_config):
    attn = CausalSelfAttention(small_config)
    x = torch.randn(2, 16, small_config.d_model)
    out = attn(x)
    assert out.shape == (2, 16, small_config.d_model)

def test_feed_forward(small_config):
    ffn = FeedForward(small_config)
    x = torch.randn(2, 16, small_config.d_model)
    out = ffn(x)
    assert out.shape == (2, 16, small_config.d_model)

def test_transformer_block(small_config):
    block = TransformerBlock(small_config)
    x = torch.randn(2, 16, small_config.d_model)
    out = block(x)
    assert out.shape == (2, 16, small_config.d_model)

def test_minillm_forward(small_config):
    model = MiniLLM(small_config)
    idx = torch.randint(0, small_config.vocab_size, (2, 16))
    logits = model(idx)
    assert logits.shape == (2, 16, small_config.vocab_size)

def test_weight_tying(small_config):
    model = MiniLLM(small_config)
    assert model.token_emb.weight is model.lm_head.weight

def test_seq_len_assertion(small_config):
    model = MiniLLM(small_config)
    idx = torch.randint(0, small_config.vocab_size, (1, small_config.seq_len + 5))
    with pytest.raises(AssertionError):
        model(idx)

