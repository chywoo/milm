import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig

class CausalSelfAttention(nn.Module):
    """하드웨어 가속(FlashAttention) 기반 Multi-Head Attention"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0, "d_model은 num_heads의 배수여야 합니다."
        
        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        
        # Q, K, V 선형 투영을 단일 행렬로 묶어 연산 효율 극대화
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape       # 32, 128, 256
        
        # (B, T, 3 * d_model) -> 3개의 (B, num_heads, T, head_dim)
        qkv = self.c_attn(x)    # qkv.shape = (32, 128, 768)
        q, k, v = qkv.chunk(3, dim=-1) # each shape = (32, 128, 256)

        # reshape to (32, 128, 8, 32) and transpose to (32, 8, 128, 32)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch 내부 최적화 C++ 커널 호출 (FlashAttention/Mem-Efficient)
        # out.shape = (32,8,128,32)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=True
        )

        # restore dimension to x.shape
        out = out.transpose(1, 2).contiguous().view(B, T, C)     # out.shape=(32,128,256)
        return self.out_proj(out)

class FeedForward(nn.Module):
    """비선형 활성화 함수(GELU)가 적용된 피드포워드 네트워크"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """Pre-LN 구조 및 Residual Connection을 포함하는 단일 블록"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN 방식: 서브레이어 진입 전 정규화, 순수 잔차 연결 유지
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MiniLLM(nn.Module):
    """전체 트랜스포머 언어 모델 (Decoder-Only)"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # 가중치 공유 (Weight Tying): Embedding과 LM Head 가중치를 공유하여 파라미터 절약 및 수렴 가속
        self.token_emb.weight = self.lm_head.weight
        
        # 파라미터 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.seq_len, f"입력 길이({T})가 최대 문맥 길이({self.cfg.seq_len})를 초과했습니다."
        
        positions = torch.arange(0, T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.drop(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_ln(x)
        logits = self.lm_head(x)
        return logits