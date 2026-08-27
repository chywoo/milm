import os
import glob
import time
import math
import logging
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx
from config import ModelConfig, TrainConfig
from model import MiniLLM
from dataset import create_dataloaders

# 표준 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_lr_scheduler(optimizer, warmup_steps, total_steps, max_lr, min_lr):
    """Cosine Annealing with Linear Warmup 스케줄러"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr / max_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(m_cfg: ModelConfig, t_cfg: TrainConfig):
    os.makedirs(t_cfg.checkpoint_dir, exist_ok=True)
    
    # 1. 텍스트 데이터 로드 (데이터 디렉토리 내의 모든 .txt 파일 로드)
    if not os.path.exists(t_cfg.data_dir):
        raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: '{t_cfg.data_dir}'. config.py의 data_dir을 확인하세요.")

    txt_files = sorted(glob.glob(os.path.join(t_cfg.data_dir, "**", "*.txt"), recursive=True))
    if not txt_files:
        raise FileNotFoundError(f"'{t_cfg.data_dir}' 디렉토리 내에 .txt 파일이 존재하지 않습니다.")

    text_list = []
    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8") as f:
            text_list.append(f.read())
    
    text = "\n".join(text_list)
    logging.info(f"데이터 로드 완료: '{t_cfg.data_dir}' 내 총 {len(txt_files)}개 .txt 파일 (총 {len(text):,} 글자)")
    
    train_loader, val_loader, tokenizer = create_dataloaders(
        text, m_cfg.seq_len, t_cfg.batch_size, t_cfg.val_split
    )
    m_cfg.vocab_size = tokenizer.vocab_size
    tokenizer.save(os.path.join(t_cfg.checkpoint_dir, "tokenizer.json"))
    logging.info(f"어휘 사전 크기: {m_cfg.vocab_size} | Train 배치 수: {len(train_loader)}")

    # 2. 모델 및 옵티마이저 초기화
    model = MiniLLM(m_cfg).to(t_cfg.device)
    if t_cfg.compile_model and t_cfg.device == "cuda":
        logging.info("PyTorch 2.0 torch.compile 활성화")
        model = torch.compile(model)
        
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=t_cfg.learning_rate, 
        weight_decay=t_cfg.weight_decay,
        fused=(t_cfg.device == "cuda")
    )
    
    total_steps = len(train_loader) * t_cfg.epochs
    scheduler = get_lr_scheduler(optimizer, t_cfg.warmup_steps, total_steps, t_cfg.learning_rate, t_cfg.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=t_cfg.use_amp and t_cfg.device == "cuda")
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    device_type = t_cfg.device if t_cfg.device in ("cuda", "mps", "cpu") else "cpu"

    # 3. 학습 루프
    logging.info(f"🚀 모델 학습 시작 (디바이스: {t_cfg.device})")
    for epoch in range(1, t_cfg.epochs + 1):
        with nvtx.range(f"Epoch_{epoch}"):
            model.train()
            train_loss = 0.0
            start_time = time.time()
            
            for step, (x, y) in enumerate(train_loader):
                with nvtx.range(f"Train_Step_{step}"):
                    with nvtx.range("H2D_Transfer"):
                        x = x.to(t_cfg.device, non_blocking=(t_cfg.device == "cuda"))
                        y = y.to(t_cfg.device, non_blocking=(t_cfg.device == "cuda"))
                    
                    optimizer.zero_grad(set_to_none=True)

                    with nvtx.range("Forward_Pass"):
                        with torch.autocast(
                            device_type=device_type, 
                            dtype=torch.float16, 
                            enabled=t_cfg.use_amp and device_type in ("cuda", "mps")
                        ):
                            logits = model(x)
                            with nvtx.range("Loss_Calculation"):
                                loss = criterion(logits.view(-1, m_cfg.vocab_size), y.view(-1))
                        
                    with nvtx.range("Backward_Pass"):
                        scaler.scale(loss).backward()

                    with nvtx.range("Optimizer_Step"):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

                    train_loss += loss.item()

            # 검증 루프
            with nvtx.range("Validation_Epoch"):
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_step, (x, y) in enumerate(val_loader):
                        with nvtx.range(f"Val_Step_{val_step}"):
                            with nvtx.range("Val_H2D_Transfer"):
                                x = x.to(t_cfg.device, non_blocking=(t_cfg.device == "cuda"))
                                y = y.to(t_cfg.device, non_blocking=(t_cfg.device == "cuda"))
                            with torch.autocast(
                                device_type=device_type, 
                                dtype=torch.float16, 
                                enabled=t_cfg.use_amp and device_type in ("cuda", "mps")
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

            # 최고 성능 모델 체크포인트 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                with nvtx.range("Save_Checkpoint"):
                    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                    ckpt_path = os.path.join(t_cfg.checkpoint_dir, t_cfg.checkpoint_name)
                    torch.save({
                        'model_state_dict': raw_model.state_dict(),
                        'model_config': m_cfg,
                        'epoch': epoch,
                        'val_loss': best_val_loss
                    }, ckpt_path)

    logging.info(f"✅ 학습 완료! 최적 모델 저장 위치: {ckpt_path}")

if __name__ == "__main__":
    train(ModelConfig(), TrainConfig())
