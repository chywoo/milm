# 🗺️ MILM 향후 개선 및 발전 과제 (Roadmap & Recommendations)

이 문서는 **MILM** 프로젝트의 성능 향상, 최신 아키텍처 도입, 추론 및 학습 최적화를 위한 단계별 개선 과제와 로드맵을 정의합니다.

---

## 📋 우선순위별 로드맵 (Roadmap Overview)

| 우선순위 | 과제명 | 구분 | 난이도 | 기대 효과 |
| :---: | :--- | :---: | :---: | :--- |
| **P0** | **Subword Tokenizer 도입 (BPE)** | 데이터/토크나이저 | 보통 | 시퀀스 길이 내 단어 정보량 3~5배 증가, 단어 분절 오류 해결 |
| **P0** | **KV Cache (Key-Value Caching)** | 추론 최적화 | 보통 | 생성 단계 시간 복잡도 $\mathcal{O}(N^2) \rightarrow \mathcal{O}(N)$ 단축 |
| **P1** | **Rotary Position Embedding (RoPE)** | 아키텍처 | 보통 | 문맥 길이 외삽(Extrapolation) 및 상대적 위치 표현력 개선 |
| **P1** | **데이터셋 로딩 및 청킹(Chunking) 최적화** | 데이터 파이프라인 | 쉬움 | 99% 중복 슬라이딩 제거, 에포크 당 학습 속도 및 메모리 개선 |
| **P2** | **RMSNorm & SwiGLU 활성화 도입** | 아키텍처 | 쉬움 | LLaMA 계열 모던 아키텍처 부합, 연산 오버헤드 감소 및 표현력 증대 |
| **P2** | **FlashAttention-2 / Triton 커널 최적화** | GPU/커널 | 보통 | 메모리 대역폭 활용 극대화 및 NVTX 기반 상세 커널 튜닝 |

---

## 🔍 상세 개선 과제

### 1. 🔡 서브워드 토크나이저 도입 (Subword Tokenizer - BPE / WordPiece)
- **현황**: 현재 `CharTokenizer`는 단일 문자(Char) 단위로 어휘(Vocab ~105)를 구성하여 `seq_len=128` 기준 실제 단어 수 20~30단어만 포함 가능.
- **개선안**:
  - Hugging Face `tokenizers` 라이브러리 기반 **Byte-Level Byte-Pair Encoding (BPE)** 또는 `tiktoken` (GPT-4 토크나이저 등) 적용.
  - 어휘 크기(`vocab_size`)를 4,096 ~ 16,384 수준으로 설정.
- **기대 효과**:
  - 동일한 `seq_len`에서 4~5배 더 긴 문맥(Context)을 한 번에 처리 가능.
  - 단어 단위 의미론적 표현력 대폭 강화 및 학습 수렴 속도 가속화.

---

### 2. ⚡ KV Cache (Key-Value Caching) 추론 엔진
- **현황**: `infer.py`에서 토큰을 하나 생성할 때마다 전체 누적 시퀀스 `input_ids[:, -seq_len:]`를 모델에 다시 입력하여 매 스텝마다 중복 $Q, K, V$를 재연산 ($\mathcal{O}(N^2)$).
- **개선안**:
  - `CausalSelfAttention`에 과거 Key와 Value 텐서를 저장해 두는 `past_key_values` (KV Cache) 버퍼 구현.
  - 다음 토큰 예측 시 방금 생성된 단일 토큰($T=1$)의 $Q$만 계산하고 기존 $K, V$에 concat하여 $\mathcal{O}(1)$ Attention 수행.
- **기대 효과**:
  - 100~200 토큰 생성 시 추론 지연 시간(Latency) 5~10배 이상 단축.

---

### 3. 🔄 회전 위치 임베딩 (RoPE - Rotary Position Embedding)
- **현황**: `nn.Embedding(seq_len, d_model)`을 통한 학습 가능한 절대 위치 임베딩(Learned Absolute Positional Embedding) 사용.
- **개선안**:
  - 현대 SOTA LLM(LLaMA, Mistral, Gemma 등)의 표준인 **RoPE (Rotary Position Embedding)** 도입.
  - 쿼리($Q$) 및 키($K$) 벡터에 회전 행렬을 적용하여 토큰 간 상대적 거리 정보를 자연스럽게 반영.
- **기대 효과**:
  - 학습 시퀀스 길이보다 긴 문맥에서도 성능 저하 없이 추론 가능한 길이 외삽(Extrapolation) 능력 확보.

---

### 4. 📦 데이터셋 청킹 및 스트라이딩 (Chunking & Packed Sequences)
- **현황**: `TextDataset`이 1글자씩 슬라이딩하여 $N - \text{seq\_len}$개의 샘플을 생성하므로 샘플 간 중복률이 99%에 달함.
- **개선안**:
  - 고정 길이(`seq_len`) 단위로 코퍼스를 청킹(Non-overlapping Chunking)하거나 `<|endoftext|>` 토큰을 사용한 Packed Sequence 파이프라인 구축.
- **기대 효과**:
  - 에포크 당 총 스텝 수를 합리적으로 줄이고 데이터 다양성을 높여 과적합(Overfitting) 방지.

---

### 5. 🛠️ 아키텍처 모던화 (RMSNorm & SwiGLU)
- **현황**: Standard LayerNorm (`nn.LayerNorm`) 및 Linear + GELU + Linear 구조의 FFN 사용.
- **개선안**:
  - **RMSNorm (Root Mean Square Layer Normalization)**: 평균 계산을 생략하고 제곱평균제곱근만으로 정규화하여 연산 속도 개선.
  - **SwiGLU (Swish Gated Linear Unit)**: $FFN(x) = (\text{Swish}(xW) \otimes xV)W_2$ 구조를 적용하여 동일 파라미터 대비 표현력 향상.

---

## 📈 장기 과제 (Long-Term Goals)

1. **Distributed Training (DDP / FSDP)**: 멀티 GPU 분산 학습 지원.
2. **SFT / DPO Fine-tuning Pipeline**: Instruction Tuning 및 인간 피드백 기반 강화학습(DPO/RLHF) 모듈 확장.
3. **ONNX / TensorRT-LLM 변환**: 에지 및 프로덕션 환경 배포를 위한 가속 런타임 지원.

