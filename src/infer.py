import os
from typing import Optional
import torch
import torch.nn.functional as F
from torch.profiler import record_function

try:
    from .config import ModelConfig
    from .model import MiniLLM
    from .dataset import CharTokenizer
except (ImportError, ValueError):
    from config import ModelConfig
    from model import MiniLLM
    from dataset import CharTokenizer

class LLMInferenceEngine:
    """Inference pipeline supporting safe sampling and decoding strategies."""
    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: Optional[str] = None):
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        # Load tokenizer and checkpoint weights
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
        """Autoregressive text generation with Top-k, Top-p (Nucleus), and repetition penalty."""
        with record_function("LLM_Generate"):
            tokens = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

            for step in range(max_new_tokens):
                with record_function(f"Generate_Step_{step}"):
                    # Crop context to maximum sequence length
                    cond_input = input_ids[:, -self.cfg.seq_len:]
                    
                    with record_function("Model_Forward"):
                        logits = self.model(cond_input)[:, -1, :] # Logits of the last token position

                    with record_function("Repetition_Penalty"):
                        # 1. Apply Repetition Penalty
                        for token_id in set(input_ids[0].tolist()):
                            if logits[0, token_id] > 0:
                                logits[0, token_id] /= repetition_penalty
                            else:
                                logits[0, token_id] *= repetition_penalty

                    with record_function("Sampling_TopK_TopP"):
                        # 2. Temperature scaling
                        logits = logits / max(temperature, 1e-5)

                        # 3. Top-k filtering
                        if top_k > 0:
                            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = float('-inf')

                        # 4. Top-p (Nucleus) filtering
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            
                            sorted_indices_to_remove = cumulative_probs > top_p
                            # Keep at least the highest probability token
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            logits[indices_to_remove] = float('-inf')

                        # 5. Sampling and token concatenation
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
        print("\n--- Generation Result ---")
        print(result)
    else:
        print("Checkpoint file not found. Please run train.py first.")
