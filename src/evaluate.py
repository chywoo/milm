import os
import math
from typing import List, Dict, Tuple, Optional
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.cuda.nvtx as nvtx

try:
    from .config import ModelConfig, TrainConfig, load_config
    from .model import MiniLLM
    from .dataset import CharTokenizer, TextDataset
except (ImportError, ValueError):
    from config import ModelConfig, TrainConfig, load_config
    from model import MiniLLM
    from dataset import CharTokenizer, TextDataset

# =====================================================================
# 1. 지표 계산 함수 (BLEU 및 ROUGE-L)
# =====================================================================
def calculate_ngram_precision(reference: List[str], candidate: List[str], n: int) -> float:
    """단일 n-gram 정밀도 계산 (문자 단위)"""
    if len(candidate) < n or len(reference) < n:
        return 0.0
    
    cand_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)]
    ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)]
    
    cand_counts = Counter(cand_ngrams)
    ref_counts = Counter(ref_ngrams)
    
    clipped_matches = sum(min(count, ref_counts[ng]) for ng, count in cand_counts.items())
    total_cand = len(cand_ngrams)
    
    return clipped_matches / total_cand if total_cand > 0 else 0.0

def compute_bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    """문자 기반 BLEU-4 점수 계산 (Brevity Penalty 포함)"""
    ref_tokens = list(reference)
    cand_tokens = list(candidate)
    
    if len(cand_tokens) == 0:
        return 0.0
    
    precisions = []
    for n in range(1, max_n + 1):
        p = calculate_ngram_precision(ref_tokens, cand_tokens, n)
        if p == 0:
            return 0.0  # 지수 감쇠 특성
        precisions.append(p)
    
    # 기하평균 계산
    geom_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    
    # 짧은 문장에 대한 페널티 (Brevity Penalty)
    bp = 1.0 if len(cand_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
    return bp * geom_mean

def compute_rouge_l(reference: str, candidate: str) -> float:
    """최장 공통 부분 수열(LCS) 기반 ROUGE-L F1 점수 계산"""
    ref_tokens = list(reference)
    cand_tokens = list(candidate)
    
    m, n = len(ref_tokens), len(cand_tokens)
    if m == 0 or n == 0:
        return 0.0
    
    # 2D DP 테이블로 LCS 길이 계산
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    lcs_len = dp[m][n]
    precision = lcs_len / n
    recall = lcs_len / m
    
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


# =====================================================================
# 2. 평가 엔진 클래스
# =====================================================================
class LLMEvaluator:
    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str = "cpu"):
        self.device = device 
        
        # 토크나이저 및 모델 체크포인트 복원
        self.tokenizer = CharTokenizer.load(tokenizer_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.cfg: ModelConfig = checkpoint['model_config']
        self.model = MiniLLM(self.cfg).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    # -------------------------------------------------------------
    # 평가 1: Perplexity (PPL) & Cross-Entropy Loss
    # -------------------------------------------------------------
    @torch.no_grad()
    def evaluate_perplexity(self, test_text: str, batch_size: int = 32) -> Tuple[float, float]:
        """테스트 데이터셋에 대한 평균 Cross-Entropy Loss 및 Perplexity 계산"""
        with nvtx.range("Eval::Perplexity"):
            token_ids = self.tokenizer.encode(test_text)
            test_ds = TextDataset(token_ids, self.cfg.seq_len)
            
            if len(test_ds) == 0:
                raise ValueError("테스트 텍스트가 시퀀스 길이(seq_len)보다 짧습니다.")
                
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            criterion = nn.CrossEntropyLoss()
            
            total_loss = 0.0
            for step, (x, y) in enumerate(test_loader):
                with nvtx.range(f"Eval_Batch_{step}"):
                    x, y = x.to(self.device), y.to(self.device)
                    with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == "cuda")):
                        logits = self.model(x)
                        loss = criterion(logits.view(-1, self.cfg.vocab_size), y.view(-1))
                    total_loss += loss.item()
                
            avg_loss = total_loss / len(test_loader)
            perplexity = math.exp(avg_loss)
            return avg_loss, perplexity

    # -------------------------------------------------------------
    # 평가 2: N-gram 정밀도 및 텍스트 유사도 (BLEU & ROUGE)
    # -------------------------------------------------------------
    @torch.no_grad()
    def evaluate_similarity_metrics(self, test_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """(프롬프트, 정답 타깃) 쌍을 바탕으로 생성 문장의 BLEU 및 ROUGE 점수 측정"""
        with nvtx.range("Eval::Similarity"):
            bleu_scores = []
            rouge_scores = []
            
            for i, (prompt, target) in enumerate(test_pairs):
                with nvtx.range(f"Eval_Pair_{i}"):
                    generated = self._generate_completion(prompt, max_new_tokens=len(target))
                    bleu_scores.append(compute_bleu(target, generated))
                    rouge_scores.append(compute_rouge_l(target, generated))
                
            avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
            avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
            
            return {"BLEU": avg_bleu, "ROUGE-L": avg_rouge}

    # -------------------------------------------------------------
    # 평가 3: 프롬프트 벤치마크 테스트베드 (Evaluation Harness)
    # -------------------------------------------------------------
    @torch.no_grad()
    def run_benchmark_suite(self, test_prompts: List[str], temperatures: List[float] = [0.2, 0.7, 1.2]) -> List[Dict]:
        """다양한 온도(Temperature) 조건에서 벤치마크 프롬프트 생성 결과 및 반복률 측정"""
        with nvtx.range("Eval::Benchmark_Suite"):
            results = []
            
            for i, prompt in enumerate(test_prompts):
                with nvtx.range(f"Prompt_Benchmark_{i}"):
                    prompt_results = {"prompt": prompt, "generations": {}}
                    for temp in temperatures:
                        with nvtx.range(f"Temp_{temp}"):
                            output = self._generate_completion(prompt, max_new_tokens=100, temperature=temp)
                            repetition_rate = self._compute_repetition_rate(output)
                            prompt_results["generations"][f"temp_{temp}"] = {
                                "text": output,
                                "repetition_rate": repetition_rate
                            }
                    results.append(prompt_results)
                
            return results

    def _generate_completion(self, prompt: str, max_new_tokens: int, temperature: float = 0.7) -> str:
        """기본 자동회귀 텍스트 생성 보조 함수"""
        with nvtx.range("Generate_Completion"):
            tokens = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            
            for _ in range(max_new_tokens):
                cond_input = input_ids[:, -self.cfg.seq_len:]
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=(self.device == "cuda")):
                    logits = self.model(cond_input)[:, -1, :]
                
                logits = logits / max(temperature, 1e-5)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat((input_ids, next_token), dim=1)
                
            # 생성된 부분만 반환
            return self.tokenizer.decode(input_ids[0, len(tokens):].tolist())

    def _compute_repetition_rate(self, text: str, n: int = 3) -> float:
        """생성된 문장의 3-gram 반복 비율 계산 (0에 가까울수록 풍부한 표현)"""
        tokens = list(text)
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        unique_ngrams = set(ngrams)
        return 1.0 - (len(unique_ngrams) / len(ngrams))


# =====================================================================
# 3. 실행 진입점
# =====================================================================
if __name__ == "__main__":
    ckpt_path = "checkpoints/best_model.pt"
    tok_path = "checkpoints/tokenizer.json"
    _, train_config = load_config("config.yaml")
    
    if not (os.path.exists(ckpt_path) and os.path.exists(tok_path)):
        print("모델 체크포인트 또는 토크나이저 파일이 존재하지 않습니다. 먼저 scripts/train.py를 실행하세요.")
        exit(1)

    evaluator = LLMEvaluator(checkpoint_path=ckpt_path, tokenizer_path=tok_path, device=train_config.device)
    
    print("\n==========================================")
    print(" 1. Perplexity (PPL) 정량 평가")
    print("==========================================")
    sample_eval_corpus = (
        "The artificial intelligence and transformer architecture revolutionized natural language processing. "
        "Self-attention allows the model to weigh the importance of different tokens dynamically."
    ) * 10
    loss, ppl = evaluator.evaluate_perplexity(sample_eval_corpus)
    print(f"Test Loss: {loss:.4f} | Perplexity: {ppl:.2f}")

    print("\n==========================================")
    print(" 2. 문장 유사도 평가 (BLEU / ROUGE-L)")
    print("==========================================")
    test_pairs = [
        ("The artificial ", "intelligence and transformer architecture"),
        ("Self-attention allows ", "the model to weigh the importance"),
    ]
    sim_scores = evaluator.evaluate_similarity_metrics(test_pairs)
    print(f"BLEU-4 Score : {sim_scores['BLEU']:.4f}")
    print(f"ROUGE-L Score: {sim_scores['ROUGE-L']:.4f}")

    print("\n==========================================")
    print(" 3. 프롬프트 벤치마크 테스트베드 결과")
    print("==========================================")
    benchmark_prompts = [
        "The transformer ",
        "Residual connections "
    ]
    bench_results = evaluator.run_benchmark_suite(benchmark_prompts, temperatures=[0.2, 0.7, 1.2])
    for item in bench_results:
        print(f"\n프롬프트: '{item['prompt']}'")
        for temp_k, gen_info in item['generations'].items():
            print(f"  [{temp_k}] (반복률: {gen_info['repetition_rate']:.2f}) -> {gen_info['text'].strip()}")
