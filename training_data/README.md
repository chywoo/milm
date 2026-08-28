# 📚 MiniLLM Pre-training Datasets Guide

본 디렉토리(`training_data/`)는 **MiniLLM (경량 트랜스포머 언어 모델)**의 효과적인 사전 학습(Pre-training)을 위해 엄선된 고품질 텍스트 코퍼스 모음입니다. 모델 크기와 학습 목적(문학적 표현, 동화/추론, 백과사전 지식, 추리 소설 등)에 따라 적절한 데이터셋을 선택하거나 조합하여 학습할 수 있습니다.

---

## 📊 수집된 데이터셋 요약 (Dataset Overview)

| 데이터셋 파일명 | 출처 (Source) | 파일 크기 | 라인 수 | 글자 수 (Chars) | 추천 학습 용도 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`tinyshakespeare.txt`** | Andrej Karpathy (char-rnn / nanoGPT) | 1.1 MB | 40,000 | ~1.11M | 빠른 프로토타이핑, 운율/희곡 생성 |
| **`tinystories_sample.txt`** | Microsoft Research (Ronen Eldan & Yuanzhi Li) | 5.0 MB | 37,063 | ~5.24M | **소형 LLM 추론/문법 학습 (최고 추천)** |
| **`wikitext2_train.txt`** | Salesforce Research / PyTorch | 10.3 MB | 36,718 | ~10.78M | 백과사전 지식, 사실적 서술, 벤치마크 |
| **`sherlock_holmes.txt`** | Project Gutenberg (Arthur Conan Doyle) | 593 KB | 12,306 | ~581K | 내러티브 연속성, 장문 추리/대화 문체 |
| **`alice_in_wonderland.txt`** | Project Gutenberg (Lewis Carroll) | 148 KB | 3,384 | ~145K | 극소형 모델 테스트, 판타지 문학 문체 |

---

## 🔍 데이터셋별 상세 소개 및 출처

### 1. Tiny Shakespeare (`tinyshakespeare.txt`)
* **출처 (Source)**: [Andrej Karpathy char-rnn GitHub Repository](https://github.com/karpathy/char-rnn)
* **다운로드 URL**: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
* **라이선스**: Public Domain / Open Source
* **소개**: 셰익스피어의 대표적인 희곡/비극 작품들의 대사와 지문으로 구성된 데이터셋입니다. Karpathy의 `nanoGPT` 및 `minGPT`에서 표준 벤치마크로 널리 사용됩니다.
* **학습 특징**:
  * 인물 간 대화(`First Citizen:`, `MENENIUS:`)와 독백 구조가 명확함
  * 약 1.1MB 크기로 단일 GPU/Apple Silicon에서 수 분 내에 수렴 확인 가능
  * 학습 후 셰익스피어 특유의 고풍스러운 어휘와 운율을 효과적으로 생성

### 2. TinyStories Sample (`tinystories_sample.txt`) ⭐ (강력 추천)
* **출처 (Source)**: Microsoft Research ([TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759))
* **다운로드 URL (HuggingFace)**: `https://huggingface.co/datasets/roneneldan/TinyStories`
* **라이선스**: CDLA-Permissive-2.0
* **소개**: GPT-3.5 및 GPT-4를 활용해 3~4세 아동 수준의 어휘로 생성한 수백만 편의 짧은 동화 데이터셋입니다. 1천만 개 미만의 극소형 파라미터 언어 모델에서도 문법적으로 완벽하고 인과관계와 상식적 추론이 담긴 텍스트를 생성하도록 설계되었습니다.
* **학습 특징**:
  * 어휘가 단순하고 명확하여 소형 토크나이저 및 모델의 수렴 속도가 매우 빠름
  * 문맥 일관성(Coherence), 인과 관계, 기본 상식 추론 학습에 최적화
  * PPL(Perplexity) 감소 추세가 매우 안정적이며 의미 있는 장문 생성 가능

### 3. WikiText-2 (`wikitext2_train.txt`)
* **출처 (Source)**: Salesforce Research ([The WikiText Long Term Dependency Language Modeling Dataset](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/))
* **다운로드 URL**: `https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt`
* **라이선스**: Creative Commons Attribution-ShareAlike License (CC BY-SA 4.0)
* **소개**: 영문 위키백과의 검증된 양질의 문서(Good and Featured Articles)에서 추출된 약 200만 단어 규모의 표준 언어 모델링 벤치마크 코퍼스입니다.
* **학습 특징**:
  * 사실적 정보, 학술/역사/과학적 설명문 중심의 격식체 텍스트
  * 문장 구조가 복잡하고 고유명사와 전문 어휘가 풍부하여 풍부한 표현력 학습 가능
  * 언어 모델 성능 평가(PPL)의 표준 벤치마크로 활용

### 4. Sherlock Holmes & Alice in Wonderland (Project Gutenberg)
* **출처 (Source)**: [Project Gutenberg](https://www.gutenberg.org/)
  * **Sherlock Holmes**: `https://www.gutenberg.org/files/1661/1661-0.txt` (아서 코난 도일)
  * **Alice in Wonderland**: `https://www.gutenberg.org/files/11/11-0.txt` (루이스 캐럴)
* **라이선스**: Public Domain
* **소개**: 영문학 고전 문학 작품의 원문 텍스트입니다.
* **학습 특징**:
  * 서사 구조, 정교한 묘사, 인물 간 추리 대화 문체 학습에 적합
  * 중소 규모 코퍼스로서 빠른 오버피팅 점검 및 문체 전이(Style Transfer) 실험에 용이

---

## 🚀 학습 데이터 사용법 (How to Train)

### 방법 1: 전체 데이터셋으로 학습 (통합 사전 학습)
`config.py`에서 `data_dir`을 `training_data`로 설정하면, 디렉토리 내의 모든 `.txt` 파일(총 약 17.8MB)을 자동으로 병합하여 대규모 사전 학습을 수행합니다.

```python
# config.py
@dataclass
class TrainConfig:
    data_dir: str = "training_data"  # 전체 데이터 로드
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 5e-4
```

```bash
python train.py
```

---

### 방법 2: 특정 데이터셋만 선택하여 학습
특정 코퍼스(예: `tinystories_sample.txt` 또는 `tinyshakespeare.txt`)만 집중 학습하고자 할 경우, 해당 파일을 별도 디렉토리(예: `data/`)에 복사하여 실행합니다.

```bash
# 1. TinyStories 동화 데이터만 학습할 경우
mkdir -p data
cp training_data/tinystories_sample.txt data/

# 2. 셰익스피어 희곡만 학습할 경우
# cp training_data/tinyshakespeare.txt data/

# 3. 학습 실행 (config.py의 data_dir="data" 기본값 사용)
python train.py
```

---

## 🎯 추천 하이퍼파라미터 가이드 (Recommended Settings)

| 데이터셋 | 모델 차원 (`d_model` / `layers`) | 문맥 길이 (`seq_len`) | 추천 `batch_size` | 추천 `epochs` | 예상 학습 시간 (Apple Silicon / RTX 3060) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tiny Shakespeare** | 256 / 6 layers | 128 | 32 ~ 64 | 50 ~ 100 | ~ 3분 |
| **TinyStories Sample** | 384 / 8 layers | 256 | 32 ~ 64 | 30 ~ 50 | ~ 8분 |
| **WikiText-2** | 512 / 8 layers | 256 | 32 ~ 64 | 20 ~ 40 | ~ 15분 |
| **전체 통합 코퍼스** | 512 / 8 layers | 256 | 64 | 20 ~ 30 | ~ 25분 |
