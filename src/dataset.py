import os
import json
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader

class CharTokenizer:
    """문자 기반 토크나이저 (어휘 사전 저장 및 불러오기 지원)"""
    def __init__(self, vocab: Optional[List[str]] = None):
        self.vocab = vocab or []
        self.char_to_idx: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char: Dict[int, str] = {i: ch for i, ch in enumerate(self.vocab)}

    @classmethod
    def fit_from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(list(set(text)))
        return cls(vocab=chars)

    def encode(self, text: str) -> List[int]:
        # 미등록 문자는 공백(0)으로 대체
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, indices: List[int]) -> str:
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'vocab': self.vocab}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(vocab=data['vocab'])

class TextDataset(Dataset):
    """자기회귀 언어 모델 학습을 위한 시퀀스 슬라이싱 데이터셋"""
    def __init__(self, token_ids: List[int], seq_len: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

def create_dataloaders(
    text_data: str, 
    seq_len: int, 
    batch_size: int, 
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, CharTokenizer]:
    """텍스트로부터 Train/Val DataLoader 및 Tokenizer 생성"""
    tokenizer = CharTokenizer.fit_from_text(text_data)
    token_ids = tokenizer.encode(text_data)
    
    split_idx = int(len(token_ids) * (1 - val_split))
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]
    
    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer

