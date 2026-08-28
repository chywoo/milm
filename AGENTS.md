# AGENTS.md

## 목적 (Purpose)
본 프로젝트의 이름은 **MILM**입니다. Lightweight Transformer Decoder-Only Language Model이며, LLM의 학습 목적으로 시작했습니다.

이 문서는 MILM 저장소에서 AI 에이전트 및 개발자가 작업할 때 준수해야 할 작업 방식과 규칙을 정의합니다.
코드 분석, 파일 수정, 환경 점검, 테스트 실행, 커밋 준비 및 변경 사항 적용 시 본 지침을 반드시 따르십시오.

특정 모듈이나 문서(예: `README.md`, `ROADMAP.md`)에 본 문서와 상충되는 구체적인 지침이 명시된 경우, 해당 상위/특화 문서를 우선 적용합니다.

---

## 개발 가이드 (Development Guidelines)

### 1. 코딩 전 깊이 생각하기 (Think Before Coding)
**추측하지 말고, 모호함을 숨기지 않으며, 트레이드오프를 명확히 제시합니다.**

구현을 시작하기 전에:
- 가정을 명시적으로 밝히십시오. 불확실한 경우 먼저 질문하십시오.
- 여러 해석이 가능한 경우 임의로 하나를 선택하지 말고 대안들을 제시하십시오.
- 더 단순한 접근 방식이 존재한다면 이를 알리고, 필요 시 이견을 제시하십시오.
- 불명확한 점이 있다면 작업을 멈추고 혼란스러운 부분을 명시하여 질문하십시오.

### 2. 단순성 우선 (Simplicity First)
**문제를 해결하는 최소한의 코드만 작성하며, 불필요하게 미래를 예측한 코드를 작성하지 않습니다.**

- 요청된 범위를 벗어나는 기능을 추가하지 마십시오.
- 단 1회만 사용되는 코드에 불필요한 추상화 계층을 만들지 마십시오.
- 요청받지 않은 "유연성"이나 "설정 가능성"을 임의로 추가하지 마십시오.
- 발생 불가능한 시나리오에 대한 과도한 예외 처리를 피하십시오.
- 200줄로 작성된 코드가 50줄로 간결화될 수 있다면 다시 작성하십시오.
- 스스로 질문하십시오: *"시니어 엔지니어가 보았을 때 과도하게 복잡하다고 느낄 것인가?"* 그렇다면 단순화하십시오.

### 3. 정밀하고 제한적인 변경 (Surgical Changes)
**필요한 부분만 수정하고, 본인이 발생시킨 부수 효과만 정리합니다.**

- **기존 코드 수정 시**:
  - 인접한 무관한 코드, 주석, 포맷을 임의로 "개선"하려 하지 마십시오.
  - 고장 나지 않은 정상 코드를 리팩토링하지 마십시오.
  - 본인의 선호 스타일과 다르더라도 프로젝트의 기존 스타일을 준수하십시오.
  - 작업과 무관한 죽은 코드(Dead Code)를 발견하면 임의로 삭제하지 말고 언급만 하십시오.
- **본인 변경으로 인해 미사용 코드가 발생한 경우**:
  - 본인의 변경으로 인해 더 이상 사용되지 않게 된 import, 변수, 함수는 깨끗이 제거하십시오.
  - 요청받지 않은 기존 레거시 미사용 코드는 삭제하지 마십시오.
- **검증 기준**: 변경된 모든 코드 라인은 사용자의 요구사항과 직접적으로 추적 가능해야 합니다.

### 4. 목표 지향적 실행 (Goal-Driven Execution)
**성공 기준을 정의하고 검증될 때까지 루프를 반복합니다.**

- 작업을 검증 가능한 목표로 구체화하십시오:
  - "유효성 검사 추가" -> "잘못된 입력에 대한 테스트 작성 후 통과시키기"
  - "버그 수정" -> "버그를 재현하는 테스트 작성 후 통과시키기"
  - "X 리팩토링" -> "리팩토링 전후 테스트가 동일하게 통과하는지 확인하기"
- 여러 단계의 작업인 경우 간결한 계획을 수립하십시오:
  ```text
  1. [단계 1] -> 검증: [검사 항목]
  2. [단계 2] -> 검증: [검사 항목]
  3. [단계 3] -> 검증: [검사 항목]
  ```
- 명확하고 엄격한 성공 기준은 자율적인 루프 실행을 가능하게 합니다. 모호한 기준("작동하게 만들기")은 불필요한 질의를 유발합니다.

---

## 환경 설정 (Environment Setup)

- **필수 로컬 도구 및 라이브러리 설치**:
  - Python 3.8 이상
  - PyTorch (CUDA, MPS 또는 CPU 가속 런타임)
  - PyYAML (`pyyaml`)
  - (선택) NVIDIA Nsight Systems (`nsys`) / Nsight Compute (`ncu`) (GPU 프로파일링용)
- **가상환경 활성화**:
  - `source .venv/bin/activate` (가상환경 사용 시)
- **승인된 실행 환경**:
  - 프로젝트 루트 및 활성화된 Python 가상환경
- **임시 스크립트 및 데이터 작업 경로**:
  - 임시 디버깅 스크립트나 검증용 파일은 워크스페이스 외부가 아닌 로컬 임시 폴더(`scratch/` 또는 `.tmp/`)에만 배치하며, 작업 종료 시 정리합니다.

### 환경 규칙 (Environment Rules)
- 프로젝트에 명시되지 않은 외부 패키지를 승인 없이 임의로 전역(`global`) 설치하지 마십시오.
- 자동 생성된 가중치 파일(`checkpoints/*.pt`)이나 대용량 데이터셋(`data/*.txt`), 프로파일링 덤프(`*.nsys-rep`)를 Git에 임의로 추가하지 마십시오.
- CI/CD 및 로컬 실행 환경과의 재현성(Reproducibility)을 최우선으로 고려합니다.

---

## 주요 명령어 (Common Commands)

### 1. 실행 및 학습 (Execution & Training)
- **모델 학습 실행**:
  ```bash
  python scripts/train.py
  ```
- **텍스트 생성 및 추론 테스트**:
  ```bash
  python scripts/infer.py --prompt "Harry looked at " --temp 0.7
  ```
- **정량/정성 성능 평가 (PPL, BLEU-4, ROUGE-L, 벤치마크)**:
  ```bash
  python scripts/evaluate.py
  ```

### 2. 단위 테스트 실행 (Tests)
- **PyTest 전체 테스트 스위트 실행**:
  ```bash
  pytest -v tests/
  ```

### 3. 설정 검증 (Configuration Check)
- **YAML 설정 로드 테스트**:
  ```bash
  python -c "from src.config import load_config; m, t = load_config('config.yaml'); print('ModelConfig:', m); print('TrainConfig:', t)"
  ```
- **설정 템플릿 복사**:
  ```bash
  cp config.yaml.template config.yaml
  ```

### 4. 정적 검사 및 코드 스타일 (Static Checks & Lint)
- **코드 포맷팅**:
  ```bash
  black .
  ```
- **린트 검사**:
  ```bash
  flake8 . --max-line-length=120
  ```
- **타입 검사**:
  ```bash
  mypy src/ tests/ scripts/
  ```

### 5. GPU 프로파일링 (Profiling Commands)
- **Nsight Systems 학습 프로파일링**:
  ```bash
  ./scripts/profile.sh nsys
  ```
- **Nsight Compute 커널 분석 (FlashAttention SDPA)**:
  ```bash
  ./scripts/profile.sh ncu
  ```

### 명령어 사용 규칙 (Command Rules)
- 변경 사항을 검증할 때는 가장 빠르고 범위가 좁은 검사부터 실행하십시오 (`pytest -v tests/`).
- 하드웨어 가속(CUDA/MPS)이 없는 환경에서는 CPU fallback이 정상 작동하는지 확인하십시오.

---

## 저장소 구조 (Repository Structure)

```text
milm/
├── src/                          # 애플리케이션 코어 패키지 (Flat 레이아웃)
│   ├── __init__.py               # 패키지 공용 모듈 익스포트
│   ├── config.py                 # ModelConfig, TrainConfig 및 YAML 설정 로더
│   ├── model.py                  # MiniLLM, TransformerBlock, CausalSelfAttention, FeedForward
│   ├── dataset.py                # CharTokenizer, TextDataset, create_dataloaders
│   ├── train.py                  # 모델 학습 코어 파이프라인
│   ├── infer.py                  # 자동회귀 텍스트 생성 추론 엔진
│   └── evaluate.py               # PPL, BLEU-4, ROUGE-L 정량/정성 평가 엔진
├── tests/                        # PyTest 기반 단위/통합 테스트 스위트
│   ├── __init__.py
│   ├── test_config.py            # 설정 로딩, YAML 파싱 및 기본값 검증
│   ├── test_model.py             # 모델 Forward Pass Shape, Weight Tying 검증
│   ├── test_dataset.py           # CharTokenizer 인코딩/디코딩, TextDataset 검증
│   └── test_evaluate.py          # BLEU-4, ROUGE-L, Perplexity 계산 검증
├── docs/                         # 프로젝트 심층 분석 및 로드맵 문서
│   ├── ROADMAP.md                # 단계별 개선 과제 및 아키텍처 로드맵
│   ├── TRAINING_REPORT.md        # 모델 학습 결과, 손실 함수 추이 및 과적합 분석 보고서
│   └── AGENTS.md                 # AI 에이전트 및 개발자를 위한 가이드라인
├── scripts/                      # 자동화, CLI 실행 및 GPU 프로파일링 유틸리티
│   ├── train.py                  # 모델 학습 CLI 런처
│   ├── infer.py                  # 텍스트 생성 추론 CLI 런처
│   ├── evaluate.py               # 성능 평가 CLI 런처
│   └── profile.sh                # Nsight Systems / Compute GPU 프로파일링 자동화 스크립트
├── checkpoints/                  # 최적 가중치(best_model.pt) 및 토크나이저 아티팩트 (Git 무시)
├── data/                         # 학습용 텍스트 코퍼스 데이터 (Git 무시)
├── config.yaml                   # 로컬 모델/학습 하이퍼파라미터 설정 파일
├── config.yaml.template          # 설정 파일 템플릿 (버전 관리 대상)
├── pyproject.toml                # 패키지 메타데이터 및 빌드 설정 (pip install -e .)
├── README.md                     # 프로젝트 메인 소개 및 가이드 문서
├── AGENTS.md                     # 프로젝트 루트 AI 에이전트 규칙
└── .gitignore                    # Git 추적 제외 규칙
```

### 구조 규칙 (Structure Rules)
- **계층 분리**: 각 모듈의 고유 역할에 맞는 코드만 해당 디렉토리/파일에 작성합니다 (`src/`, `tests/`, `docs/`, `scripts/`).
- **모듈 간 의존성 방향 준수**: 하위 기반 모듈(`src/config.py`, `src/model.py`, `src/dataset.py`)이 상위 실행 런처(`scripts/`)를 역참조하지 않도록 합니다.
- **새로운 파일 생성 시**: 성격에 맞게 `src/`, `tests/`, `docs/`, `scripts/` 중 적절한 위치에 배치하고 `README.md` 및 `AGENTS.md`에 반영합니다.

---

## 아키텍처 경계 및 원칙 (Architecture Boundaries)

1. **`src/config.py`**:
   - 모델 및 학습 하이퍼파라미터의 데이터 구조(`dataclass`) 및 `config.yaml` I/O를 전담합니다.
   - 신경망 가중치 연산이나 학습 루프에 직접 관여하지 않습니다.
2. **`src/model.py`**:
   - `MiniLLM`, `TransformerBlock`, `CausalSelfAttention`, `FeedForward` 등 순수 PyTorch 신경망 모듈만 정의합니다.
   - 디스크 파일 로드, 데이터셋 파싱, 체크포인트 저장 로직을 포함하지 않습니다.
   - 모든 주요 연산 구간에 일관된 `torch.cuda.nvtx.range` 마커를 유지해야 합니다.
3. **`src/dataset.py`**:
   - 토큰화(`CharTokenizer`) 및 데이터로더 생성(`create_dataloaders`)에만 책임을 갖습니다.
   - 모델의 전방 연산이나 손실 함수 계산을 수행하지 않습니다.
4. **`src/train.py` / `src/infer.py` / `src/evaluate.py` / `scripts/`**:
   - `config`, `model`, `dataset` 계층을 조합하여 학습, 추론, 평가 파이프라인을 실행합니다.
   - 모델 내부의 가중치 텐서에 직접 비인가 수정을 가하지 않고 공용 인터페이스를 통해 접근합니다.

### 아키텍처 규칙
- 레이어 간 우회 호출이나 단축 코드를 도입하지 마십시오.
- 아키텍처 규칙을 수정해야 하는 대규모 리팩토링의 경우 사전에 설계 검토를 거쳐야 합니다.

---

## 보안 및 데이터 무결성 모델 (Security & Integrity Model)

### 보안 규칙 (Security Rules)
- **자격 증명 보호**: API 키, 개인 토큰, 비밀번호 등 민감 정보를 코드나 커밋에 절대 포함하지 마십시오.
- **안전한 직렬화/역직렬화**:
  - `torch.load` 시 신뢰할 수 없는 외부 가중치를 무단 로드하지 않도록 주의합니다.
  - YAML 파싱 시 반드시 `yaml.safe_load`를 사용하십시오.
- **파일 경로 검증**:
  - `data_dir`, `checkpoint_dir` 등 파일 경로 처리 시 디렉토리 순회(`Directory Traversal`, `../`) 취약점이 발생하지 않도록 정규화된 경로를 사용하십시오.

---

## 코딩 표준 (Coding Standards)

### 일반 규칙
- **기존 스타일 유지**: 간결하고 명확한 Python 표준(PEP 8) 코딩 스타일을 유지합니다.
- **명확한 네이밍**: 변수, 함수, 클래스명에 역할을 직관적으로 드러내는 명확한 이름을 사용합니다.
- **타입 힌트**: 모든 주요 함수 시그니처 및 클래스 필드에 `typing` (`Tuple`, `List`, `Dict`, `Optional` 등)을 명시합니다.
- **Docstring 및 주석**: 비직관적인 알고리즘이나 텐서 차원 변경 구간(`transpose`, `view`, `chunk` 등)에는 반드시 주석과 shape 설명을 첨부합니다.
- **프로파일러 계층 구조 준수**: 모델 포워드, 역전파, H2D 전송, 추론 루프의 계층적 `torch.profiler.record_function` 마커를 훼손하지 않습니다.

---

## 테스트 및 평가 표준 (Testing Standards)

- **새로운 기능 추가 시 검증 필수**:
  - 새로운 아키텍처 모듈(예: RoPE, RMSNorm, KV Cache) 또는 토크나이저 변경 시 독립 단위 동작 검증을 수행합니다.
- **평가 지표 유지 및 비교**:
  - 모델 수정 후 [`evaluate.py`](evaluate.py)를 실행하여 Cross-Entropy Loss, Perplexity (PPL), BLEU-4, ROUGE-L 지표가 정상 산출되는지 확인합니다.
- **정상/경계/예외 케이스 검증**:
  - 입력 시퀀스 길이 초과(`T > seq_len`), 알 수 없는 문자 입력, 빈 문자열 입력 등에 대한 방어 로직을 점검합니다.
