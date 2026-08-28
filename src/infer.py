import os
from typing import Optional
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

try:
    from .config import ModelConfig
    from .model import MiniLLM
    from .dataset import CharTokenizer
except (ImportError, ValueError):
    from config import ModelConfig
    from model import MiniLLM
    from dataset import CharTokenizer

class LLMInferenceEngine:
    """안전한 샘플링과 디코딩을 지원하는 추론 파이프라인"""
    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: Optional[str] = None):
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        # 토크나이저 및 가중치 복원
        self.tokenizer = CharTokenizer.load(tokenizer_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.cfg: ModelConfig = checkpoint['model_config']
        self.model = MiniLLM(self.cfg).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 200, 
        temperature: float = 0.7, 
        top_k: int = 10,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> str:
        """Top-k, Top-p(Nucleus), 반복 패널티가 적용된 자동회귀 생성"""
        with nvtx.range("LLM_Generate"):
            tokens = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

            for step in range(max_new_tokens):
                with nvtx.range(f"Generate_Step_{step}"):
                    # 문맥 크기 유지
                    cond_input = input_ids[:, -self.cfg.seq_len:]
                    
                    with nvtx.range("Model_Forward"):
                        logits = self.model(cond_input)[:, -1, :] # 마지막 위치 토큰 로짓

                    with nvtx.range("Repetition_Penalty"):
                        # 1. 반복 페널티 적용 (Repetition Penalty)
                        for token_id in set(input_ids[0].tolist()):
                            if logits[0, token_id] > 0:
                                logits[0, token_id] /= repetition_penalty
                            else:
                                logits[0, token_id] *= repetition_penalty

                    with nvtx.range("Sampling_TopK_TopP"):
                        # 2. 온도 조절 (Temperature)
                        logits = logits / max(temperature, 1e-5)

                        # 3. Top-k 필터링
                        if top_k > 0:
                            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = float('-inf')

                        # 4. Top-p (Nucleus) 필터링
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            
                            sorted_indices_to_remove = cumulative_probs > top_p
                            # 첫 번째 유효 토큰은 유지
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            logits[indices_to_remove] = float('-inf')

                        # 5. 샘플링 및 토큰 결합
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        input_ids = torch.cat((input_ids, next_token), dim=1)

            return self.tokenizer.decode(input_ids[0].tolist())

if __name__ == "__main__":
    ckpt = "checkpoints/best_model.pt"
    tok = "checkpoints/tokenizer.json"
    
    if os.path.exists(ckpt) and os.path.exists(tok):
        engine = LLMInferenceEngine(checkpoint_path=ckpt, tokenizer_path=tok)
        prompt_text = "The artificial "
        result = engine.generate(prompt=prompt_text, temperature=0.7, top_k=5, top_p=0.9)
        print("\n--- 추론 결과 ---")
        print(result)
    else:
        print("체크포인트 파일이 존재하지 않습니다. 먼저 train.py를 실행하세요.")

