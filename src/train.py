import os
import glob
import time
import math
import logging
from typing import Optional, Callable
import torch
import torch.nn as nn
from torch.profiler import record_function

try:
    from .config import ModelConfig, TrainConfig, load_config
    from .model import MiniLLM
    from .dataset import create_dataloaders
except (ImportError, ValueError):
    from config import ModelConfig, TrainConfig, load_config
    from model import MiniLLM
    from dataset import create_dataloaders

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_lr_scheduler(optimizer, warmup_steps, total_steps, max_lr, min_lr):
    """Cosine Annealing with Linear Warmup learning rate scheduler."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr / max_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(m_cfg: ModelConfig, t_cfg: TrainConfig, on_step_end: Optional[Callable[[int], None]] = None):
    os.makedirs(t_cfg.checkpoint_dir, exist_ok=True)
    
    # 1. Load text datasets (all .txt files in data directory)
    if not os.path.exists(t_cfg.data_dir):
        raise FileNotFoundError(f"Data directory not found: '{t_cfg.data_dir}'. Please check data_dir in config.yaml.")

    txt_files = sorted(glob.glob(os.path.join(t_cfg.data_dir, "**", "*.txt"), recursive=True))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{t_cfg.data_dir}' directory.")

    text_list = []
    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8") as f:
            text_list.append(f.read())
    
    text = "\n".join(text_list)
    logging.info(f"Data loading complete: {len(txt_files)} .txt file(s) in '{t_cfg.data_dir}' ({len(text):,} total characters)")
    
    train_loader, val_loader, tokenizer = create_dataloaders(
        text, m_cfg.seq_len, t_cfg.batch_size, t_cfg.val_split
    )
    m_cfg.vocab_size = tokenizer.vocab_size
    tokenizer.save(os.path.join(t_cfg.checkpoint_dir, "tokenizer.json"))
    logging.info(f"Vocabulary size: {m_cfg.vocab_size} | Train batches: {len(train_loader)}")

    # 2. Initialize model and optimizer
    model = MiniLLM(m_cfg).to(t_cfg.device)
    if t_cfg.compile_model and t_cfg.device == "cuda":
        logging.info("PyTorch 2.0 torch.compile enabled")
        model = torch.compile(model)
        
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=t_cfg.learning_rate, 
        weight_decay=t_cfg.weight_decay,
        fused=(t_cfg.device == "cuda")
    )
    
    total_steps = len(train_loader) * t_cfg.epochs
    scheduler = get_lr_scheduler(optimizer, t_cfg.warmup_steps, total_steps, t_cfg.learning_rate, t_cfg.min_lr)

    # Initialize GradScaler for AMP across PyTorch versions
    use_scaler = t_cfg.use_amp and t_cfg.device == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    else:
        class _DummyScaler:
            def scale(self, loss):
                return loss
            def unscale_(self, optimizer):
                pass
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass
        scaler = _DummyScaler()

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    device_type = t_cfg.device if t_cfg.device in ("cuda", "mps", "cpu") else "cpu"

    # 3. Training loop
    logging.info(f"Starting model training (device: {t_cfg.device})")
    for epoch in range(1, t_cfg.epochs + 1):
        with record_function(f"Epoch_{epoch}"):
            model.train()
            train_loss = 0.0
            start_time = time.time()
            
            for step, (x, y) in enumerate(train_loader):
                with record_function(f"Train_Step_{step}"):
                    with record_function("H2D_Transfer"):
                        x = x.to(t_cfg.device, non_blocking=(t_cfg.device == "cuda"))
                        y = y.to(t_cfg.device, non_blocking=(t_cfg.device == "cuda"))
                    
                    optimizer.zero_grad(set_to_none=True)

                    with record_function("Forward_Pass"):
                        with torch.autocast(
                            device_type="cuda" if t_cfg.device == "cuda" else "cpu", 
                            dtype=torch.float16, 
                            enabled=t_cfg.use_amp and device_type in ("cuda")
                        ):
                            logits = model(x)
                            with record_function("Loss_Calculation"):
                                loss = criterion(logits.view(-1, m_cfg.vocab_size), y.view(-1))
                        
                    with record_function("Backward_Pass"):
                        scaler.scale(loss).backward()

                    with record_function("Optimizer_Step"):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

                    train_loss += loss.item()
                    if on_step_end is not None:
                        on_step_end(step)

            # Validation loop
            with record_function("Validation_Epoch"):
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_step, (x, y) in enumerate(val_loader):
                        with record_function(f"Val_Step_{val_step}"):
                            with record_function("Val_H2D_Transfer"):
                                x = x.to(t_cfg.device, non_blocking=(t_cfg.device == "cuda"))
                                y = y.to(t_cfg.device, non_blocking=(t_cfg.device == "cuda"))
                            with torch.autocast(
                                device_type="cuda" if t_cfg.device == "cuda" else "cpu",  
                                dtype=torch.float16, 
                                enabled=t_cfg.use_amp and device_type in ("cuda")
                            ):
                                logits = model(x)
                                loss = criterion(logits.view(-1, m_cfg.vocab_size), y.view(-1))
                            val_loss += loss.item()
                    
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / max(1, len(val_loader))
            elapsed = time.time() - start_time
            
            if epoch % 10 == 0 or epoch == 1:
                logging.info(
                    f"Epoch [{epoch:03d}/{t_cfg.epochs:03d}] | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.2f}s"
                )

            # Save best checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                with record_function("Save_Checkpoint"):
                    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                    ckpt_path = os.path.join(t_cfg.checkpoint_dir, t_cfg.checkpoint_name)
                    torch.save({
                        'model_state_dict': raw_model.state_dict(),
                        'model_config': m_cfg,
                        'epoch': epoch,
                        'val_loss': best_val_loss
                    }, ckpt_path)

    logging.info(f"Training complete! Best checkpoint saved at: {ckpt_path}")

if __name__ == "__main__":
    m_cfg, t_cfg = load_config("config.yaml")
    train(m_cfg, t_cfg)
