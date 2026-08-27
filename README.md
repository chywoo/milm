# MiniLLM (Lightweight Transformer Decoder-Only Language Model)

PyTorch 기반의 경량 디코더 전용(Decoder-Only) 트랜스포머 언어 모델 구현체입니다. 현대적인 LLM 아키텍처(Pre-LN, Scaled Dot-Product Attention/FlashAttention, Weight Tying 등)와 학습/추론 최적화 기법, 포괄적인 평가 지표 체계(PPL, BLEU, ROUGE-L, 다중 온도 벤치마크)를 포함하고 있습니다.

---

## 주요 특징 (Key Features)

- **현대적 트랜스포머 아키텍처**:
  - Pre-Layer Normalization (Pre-LN) 및 Residual Connection 구조
  - `torch.nn.functional.scaled_dot_product_attention`을 통한 C++/하드웨어 가속 Attention
  - 입력 임베딩(Token Embedding)과 출력 투영층(LM Head) 간 **Weight Tying** 적용
  - GELU 활성화 함수 기반 FeedForward Network (FFN)
- **학습 파이프라인 최적화**:
  - Cosine Annealing with Linear Warmup 학습률 스케줄러
  - Automatic Mixed Precision (`torch.autocast` / `GradScaler`) 지원
  - `torch.compile` (PyTorch 2.0+) 및 Fused AdamW 지원
  - Gradient Clipping을 통한 기울기 폭주 방지
- **고급 디코딩 & 추론 엔진**:
  - Temperature Scaling, Top-K Filtering, Top-P (Nucleus) Filtering 지원
  - 생성 품질 개선을 위한 Repetition Penalty (반복 페널티) 내장
- **포괄적인 정량/정성 평가 도구**:
  - Cross-Entropy Loss & Perplexity (PPL) 계산
  - n-gram 기반 BLEU-4 및 LCS 기반 ROUGE-L 유사도 평가
  - 다중 Temperature 설정에 따른 텍스트 생성 및 n-gram 반복률(Repetition Rate) 분석 벤치마크

---

## 디렉토리 구조 (Directory Structure)

```text
minillm/
├── config.py       # 모델(ModelConfig) 및 학습/환경(TrainConfig) 하이퍼파라미터 설정
├── model.py        # 트랜스포머(Decoder-Only MiniLLM) 모델 아키텍처 구현
├── dataset.py      # 문자 기반 토크나이저(CharTokenizer) 및 시퀀스 데이터셋/데이터로더 파이프라인
├── train.py        # AMP, 스케줄러, 최적화 및 모델 체크포인트 저장 학습 루프
├── infer.py        # Top-k/Top-p/온도 조절/반복 페널티 기반 고속 텍스트 생성 추론 엔진
├── evaluate.py     # PPL, BLEU, ROUGE-L, 벤치마크 스위트 정량/정성 평가 모듈
└── struct.txt      # 프로젝트 구조 요약 메모
```

---

## 모듈별 상세 설명

| 파일명 | 주요 클래스 및 함수 | 설명 |
| :--- | :--- | :--- |
| `config.py` | `ModelConfig`, `TrainConfig` | 임베딩 차원(`d_model`), 레이어 수(`num_layers`), 문맥 길이(`seq_len`), 배치 크기, 학습률, AMP/Compile 설정 등을 Dataclass로 관리 |
| `model.py` | `MiniLLM`, `TransformerBlock`, `CausalSelfAttention`, `FeedForward` | Pre-LN 및 Fast Attention이 적용된 Decoder-Only 트랜스포머 언어 모델 |
| `dataset.py` | `CharTokenizer`, `TextDataset`, `create_dataloaders` | 문자 단위 어휘 사전 생성/저장/로드 및 슬라이딩 윈도우 기반 Next-Token Prediction 데이터셋 구성 |
| `train.py` | `train`, `get_lr_scheduler` | Cosine Annealing 스케줄링, FP16 AMP 학습, 검증 Loss 기반 최적 체크포인트(`checkpoints/best_model.pt`) 저장 |
| `infer.py` | `LLMInferenceEngine` | 체크포인트 로드 후 Top-k, Top-p, Repetition Penalty를 적용한 텍스트 생성 파이프라인 |
| `evaluate.py` | `LLMEvaluator`, `compute_bleu`, `compute_rouge_l` | 모델 검증을 위한 Perplexity(PPL), BLEU-4, ROUGE-L, 다중 온도 프롬프트 벤치마크 및 n-gram 반복률 분석 |

---

## 시작하기 (Getting Started)

### 1. 환경 준비
Python 3.8 이상 및 PyTorch가 설치되어 있어야 합니다.

```bash
# 가상환경 활성화 (필요 시)
source .venv/bin/activate
```

### 2. 모델 학습 (Training)
```bash
python train.py
```
- 학습이 완료되면 `checkpoints/` 디렉토리에 최적 모델 체크포인트(`best_model.pt`)와 어휘 사전(`tokenizer.json`)이 저장됩니다.

### 3. 텍스트 추론 및 생성 (Inference)
```bash
python infer.py
```
- 프롬프트를 입력받아 자기회귀(Autoregressive) 방식으로 다음 토큰들을 샘플링하여 텍스트를 완성합니다.

### 4. 모델 성능 평가 (Evaluation)
```bash
python evaluate.py
```
- Perplexity (PPL), BLEU, ROUGE-L 지표 및 Temperature별 생성 결과를 종합 평가합니다.

---

## 기본 설정 (Default Hyperparameters)

```python
# ModelConfig
vocab_size = 256   # 데이터 기반 동적 결정
seq_len    = 128   # 최대 문맥 길이
d_model    = 256   # 임베딩 / 히든 차원
num_heads  = 8     # Attention Head 수
num_layers = 6     # Transformer Block 레이어 수
d_ff       = 1024  # FFN 내부 확장 차원 (4 * d_model)
dropout    = 0.1

# TrainConfig
batch_size    = 32
epochs        = 100
learning_rate = 5e-4
warmup_steps  = 100
weight_decay  = 0.01
grad_clip     = 1.0
use_amp       = True
```

---

## ⚡ NVIDIA 프로파일링 (NVIDIA NVTX & Nsight Profiling)

코드베이스 전반에 **NVIDIA Tools Extension (NVTX)** 마커가 내장되어 있어, **NVIDIA Nsight Systems (`nsys`)** 및 **Nsight Compute (`ncu`)**를 사용하여 CUDA 커널 연산, H2D/D2H 메모리 복사 병목, 레이어별 연산 소요 시간을 시각적으로 정밀 분석할 수 있습니다.

### 1. NVTX 계층 구조 (NVTX Range Hierarchy)

* **모델 아키텍처 (`model.py`)**:
  * `MiniLLM::forward`
    * `Embedding_PosEncoding`: 임베딩 및 Positional Encoding 연산
    * `Block_{0..N}`: 각 트랜스포머 블록 레이어
      * `PreLN1_SelfAttention` -> `CausalSelfAttention` (`QKV_Projection`, `QKV_Reshape`, `FlashAttention_SDPA`, `Out_Projection`)
      * `PreLN2_FeedForward` -> `FeedForward`
    * `Final_LayerNorm`: 최종 레이어 정규화
    * `LM_Head`: 최종 로짓 프로젝션
* **학습 파이프라인 (`train.py`)**:
  * `Epoch_{i}` -> `Train_Step_{step}`
    * `H2D_Transfer`: CPU to GPU 텐서 전송
    * `Forward_Pass` / `Loss_Calculation`
    * `Backward_Pass`: 역전파 기울기 계산
    * `Optimizer_Step`: Grad Scaler 언스케일링, Gradient Clipping 및 가중치 업데이트
  * `Validation_Epoch` -> `Val_Step_{step}` (`Val_H2D_Transfer`, Forward)
  * `Save_Checkpoint`: 모델 가중치 직렬화
* **추론 및 평가 (`infer.py`, `evaluate.py`)**:
  * `LLM_Generate` -> `Generate_Step_{step}` (`Model_Forward`, `Repetition_Penalty`, `Sampling_TopK_TopP`)
  * `Eval::Perplexity`, `Eval::Similarity`, `Eval::Benchmark_Suite`

---

### 2. 프로파일링 실행 방법

#### 🔹 NVIDIA Nsight Systems (`nsys`) 프로파일링
전체 시스템 타임라인(CUDA 커널, NVTX 범위, 메모리 복사, CPU OS 런타임)을 프로파일링합니다.

```bash
# 1. 학습 루프 프로파일링
nsys profile \
  -t cuda,nvtx,osrt \
  -s cpu \
  --output=profile_train \
  --export=sqlite \
  python train.py

# 2. 추론 파이프라인 프로파일링
nsys profile \
  -t cuda,nvtx,osrt \
  --output=profile_infer \
  python infer.py

# 3. 특정 NVTX 범위(예: 1번 에포크)만 타겟팅하여 캡처
nsys profile \
  -t cuda,nvtx \
  -c nvtx \
  -p "Epoch_1@*" \
  --output=profile_epoch1 \
  python train.py
```

#### 🔹 NVIDIA Nsight Compute (`ncu`) 커널 정밀 분석
특정 NVTX 범위 내의 GPU 커널(예: FlashAttention / SDPA) 성능 및 메모리 대역폭을 상세 분석합니다.

```bash
# FlashAttention SDPA 커널 정밀 분석
ncu --nvtx --nvtx-include "FlashAttention_SDPA" \
  --set full \
  -o profile_sdpa_kernel \
  python train.py
```

#### 🔹 결과 시각화 (GUI)
1. 생성된 `profile_train.nsys-rep` 파일을 로컬 머신으로 다운로드합니다.
2. **NVIDIA Nsight Systems GUI** 애플리케이션에서 열어 `NVTX` 타임라인 레인을 확장하면 계층별 실행 시간과 GPU 병목 구간을 확인할 수 있습니다.
