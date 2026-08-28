import os
import tempfile
import pytest
import torch
from src.dataset import CharTokenizer, TextDataset, create_dataloaders

def test_char_tokenizer_basic():
    sample_text = "hello world"
    tok = CharTokenizer.fit_from_text(sample_text)
    
    assert tok.vocab_size == len(set(sample_text))
    encoded = tok.encode("hello")
    decoded = tok.decode(encoded)
    assert decoded == "hello"

def test_char_tokenizer_unknown_chars():
    sample_text = "abc"
    tok = CharTokenizer.fit_from_text(sample_text)
    encoded = tok.encode("abcd")  # 'd' is unknown
    assert len(encoded) == 4
    assert encoded[3] == 0  # fallback to 0

def test_char_tokenizer_save_and_load():
    sample_text = "abc 123 !?#"
    tok = CharTokenizer.fit_from_text(sample_text)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tok_file = os.path.join(tmpdir, "tokenizer.json")
        tok.save(tok_file)
        
        loaded_tok = CharTokenizer.load(tok_file)
        assert loaded_tok.vocab == tok.vocab
        assert loaded_tok.encode("abc") == tok.encode("abc")

def test_text_dataset():
    tokens = list(range(20))
    seq_len = 5
    ds = TextDataset(tokens, seq_len)
    
    assert len(ds) == 20 - 5
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2, 3, 4]
    assert y.tolist() == [1, 2, 3, 4, 5]

def test_create_dataloaders():
    sample_text = "The quick brown fox jumps over the lazy dog." * 10
    train_loader, val_loader, tokenizer = create_dataloaders(
        text_data=sample_text,
        seq_len=16,
        batch_size=4,
        val_split=0.2
    )
    
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert tokenizer.vocab_size > 0
    
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (4, 16)
    assert batch_y.shape == (4, 16)

