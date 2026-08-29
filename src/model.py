import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
from torch.profiler import record_function

try:
    from .config import ModelConfig, load_config
except (ImportError, ValueError):
    from config import ModelConfig, load_config

class CausalSelfAttention(nn.Module):
    """Hardware-accelerated (FlashAttention / SDPA) Multi-Head Attention."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0, "d_model must be divisible by num_heads."
        
        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        
        # Fuse Q, K, V linear projections into a single matrix for efficiency
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("CausalSelfAttention"):
            B, T, C = x.shape       # 32, 128, 256
            
            with record_function("QKV_Projection"):
                # (B, T, 3 * d_model) -> 3 distinct tensors of (B, num_heads, T, head_dim)
                qkv = self.c_attn(x)    # qkv.shape = (32, 128, 768)
                q, k, v = qkv.chunk(3, dim=-1) # each shape = (32, 128, 256)

            with record_function("QKV_Reshape"):
                # reshape to (32, 128, 8, 32) and transpose to (32, 8, 128, 32)
                q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            with record_function("FlashAttention_SDPA"):
                # Call PyTorch optimized C++ kernel (FlashAttention / Memory-Efficient Attention)
                out = F.scaled_dot_product_attention(
                    q, k, v, 
                    dropout_p=self.dropout if self.training else 0.0, 
                    is_causal=True
                )

            with record_function("Out_Projection"):
                # restore dimension to x.shape
                out = out.transpose(1, 2).contiguous().view(B, T, C)     # out.shape=(32,128,256)
                res = self.out_proj(out)
            return res

class FeedForward(nn.Module):
    """FeedForward Network with GELU non-linear activation."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("FeedForward"):
            return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer block with Pre-LN architecture and residual connections."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: normalize before sub-layers to maintain clean residual paths
        with record_function("TransformerBlock"):
            with record_function("PreLN1_SelfAttention"):
                x = x + self.attn(self.ln1(x))
            with record_function("PreLN2_FeedForward"):
                x = x + self.ffn(self.ln2(x))
            return x

class MiniLLM(nn.Module):
    """Full Transformer Decoder-Only Language Model."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # Weight Tying: share weights between Token Embedding and LM Head
        self.token_emb.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        with record_function("MiniLLM::forward"):
            B, T = idx.shape
            assert T <= self.cfg.seq_len, f"Input sequence length ({T}) exceeds maximum context length ({self.cfg.seq_len})."
            
            with record_function("Embedding_PosEncoding"):
                positions = torch.arange(0, T, device=idx.device)
                x = self.token_emb(idx) + self.pos_emb(positions)
                x = self.drop(x)
            
            for i, block in enumerate(self.blocks):
                with record_function(f"Block_{i}"):
                    x = block(x)
                
            with record_function("Final_LayerNorm"):
                x = self.final_ln(x)

            with record_function("LM_Head"):
                logits = self.lm_head(x)

            return logits


if __name__ == "__main__":
    m_cfg, t_cfg = load_config("config.yaml")
    model = MiniLLM(m_cfg).to(t_cfg.device)
    dummy_idx = torch.randint(0, m_cfg.vocab_size, (2, m_cfg.seq_len), device=t_cfg.device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if t_cfg.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    use_nvtx = m_cfg.emit_nvtx and torch.cuda.is_available()
    with torch.autograd.profiler.emit_nvtx(enabled=use_nvtx):
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            out = model(dummy_idx)

    print(f"MiniLLM forward pass successful: output_shape={out.shape}, emit_nvtx={m_cfg.emit_nvtx} (active={use_nvtx})")

