# AGENTS.md

## Purpose
The project name is **MILM** (Lightweight Transformer Decoder-Only Language Model), built for LLM study, experimentation, and research.

This document defines workflows, operational boundaries, coding practices, and guidelines for AI agents and developers working on the MILM repository.
Follow these instructions during code analysis, file modification, environment checks, testing, and pull request preparation.

If specific modules or documentation (e.g., `README.md`, `ROADMAP.md`) state more specific rules, those specialized documents take precedence.

---

## Development Guidelines

### 1. Think Before Coding
**Do not guess, do not hide ambiguity, and present trade-offs clearly.**

Before implementing changes:
- State assumptions explicitly. If anything is uncertain, ask first.
- If multiple interpretations exist, present alternatives rather than choosing arbitrarily.
- If a simpler approach exists, advise accordingly and propose alternatives.
- If something is unclear, pause work and ask focused questions.

### 2. Simplicity First
**Write only the minimum code necessary to solve the problem; avoid speculative engineering.**

- Do not add features outside the requested scope.
- Avoid introducing unnecessary abstraction layers for single-use code.
- Do not add unrequested "flexibility" or "configurability."
- Avoid excessive error handling for impossible scenarios.
- If a 200-line solution can be written cleanly in 50 lines, refactor to 50 lines.
- Ask yourself: *"Would a senior engineer find this overly complex?"* If yes, simplify it.

### 3. Surgical Changes
**Modify only what is needed, and clean up side effects caused by your changes.**

- **When editing existing code**:
  - Do not arbitrarily "improve" unrelated adjacent code, comments, or formatting.
  - Do not refactor functional code without a clear reason.
  - Respect existing repository style even if it differs from personal preference.
  - If you encounter dead code unrelated to the task, mention it rather than deleting it without prompt.
- **When your changes cause unused code**:
  - Remove imports, variables, or functions rendered obsolete by your changes.
  - Do not remove legacy unused code unrelated to your task.
- **Verification Criterion**: Every line changed must trace back directly to user requirements.

### 4. Goal-Driven Execution
**Define success criteria and iterate through verification loops until satisfied.**

- Formalize tasks into verifiable goals:
  - "Add validation" -> "Write a test for invalid input and make it pass."
  - "Fix a bug" -> "Write a reproduction test and make it pass."
  - "Refactor X" -> "Verify all tests continue to pass before and after refactoring."
- For multi-step tasks, establish a concise plan:
  ```text
  1. [Step 1] -> Verification: [Check item]
  2. [Step 2] -> Verification: [Check item]
  3. [Step 3] -> Verification: [Check item]
  ```
- Clear and strict criteria enable autonomous verification loops.

---

## Environment Setup

- **Required Local Tools & Libraries**:
  - Python 3.8 or higher
  - PyTorch (CUDA, MPS, or CPU acceleration runtime)
  - PyYAML (`pyyaml`)
  - (Optional) NVIDIA Nsight Systems (`nsys`) / Nsight Compute (`ncu`) (for GPU profiling)
- **Virtual Environment Activation**:
  - `source .venv/bin/activate` (when using a virtual environment)
- **Approved Execution Environment**:
  - Project root and active Python virtual environment
- **Temporary Scripts & Data**:
  - Place temporary debug scripts or scratch files in local temporary directories (`scratch/` or `.tmp/`), and clean them up when done.

### Environment Rules
- Do not install unauthorized external packages globally without confirmation.
- Do not commit generated weight checkpoints (`checkpoints/*.pt`), large datasets (`data/*.txt`), or profiling dumps (`*.nsys-rep`) to Git.
- Prioritize reproducibility across CI/CD and local environments.

---

## Common Commands

### 1. Execution & Training
- **Train the model**:
  ```bash
  python scripts/train.py
  ```
- **Text generation and inference**:
  ```bash
  python scripts/infer.py --prompt "Harry looked at " --temp 0.7
  ```
- **Quantitative & qualitative evaluation (PPL, BLEU-4, ROUGE-L, Benchmark)**:
  ```bash
  python scripts/evaluate.py
  ```

### 2. Unit Tests
- **Run PyTest test suite**:
  ```bash
  pytest -v tests/
  ```

### 3. Configuration Check
- **Validate YAML configuration loading**:
  ```bash
  python -c "from src.config import load_config; m, t = load_config('config.yaml'); print('ModelConfig:', m); print('TrainConfig:', t)"
  ```
- **Copy configuration template**:
  ```bash
  cp config.yaml.template config.yaml
  ```

### 4. Static Checks & Code Formatting
- **Code formatting**:
  ```bash
  black .
  ```
- **Linting**:
  ```bash
  flake8 . --max-line-length=120
  ```
- **Type checking**:
  ```bash
  mypy src/ tests/ scripts/
  ```

### 5. GPU Profiling
- **Nsight Systems training trace**:
  ```bash
  ./scripts/profile.sh nsys
  ```
- **Nsight Compute kernel analysis (FlashAttention SDPA)**:
  ```bash
  ./scripts/profile.sh ncu
  ```

### Command Rules
- When verifying changes, start with the fastest and most focused checks first (`pytest -v tests/`).
- Verify CPU fallback works properly in environments without hardware acceleration (CUDA/MPS).

---

## Repository Structure

```text
milm/
├── src/                          # Core application package (flat layout)
│   ├── __init__.py               # Package export module
│   ├── config.py                 # ModelConfig, TrainConfig & YAML loader
│   ├── model.py                  # MiniLLM, TransformerBlock, CausalSelfAttention, FeedForward
│   ├── dataset.py                # CharTokenizer, TextDataset, create_dataloaders
│   ├── train.py                  # Core training pipeline
│   ├── infer.py                  # Autoregressive generation inference engine
│   └── evaluate.py               # PPL, BLEU-4, ROUGE-L evaluation engine
├── tests/                        # PyTest unit and integration test suite
│   ├── __init__.py
│   ├── test_config.py            # Configuration loading & YAML verification
│   ├── test_model.py             # Forward pass shapes & weight tying verification
│   ├── test_dataset.py           # CharTokenizer encoding/decoding & dataset verification
│   └── test_evaluate.py          # BLEU-4, ROUGE-L & Perplexity computation verification
├── docs/                         # In-depth analysis & roadmap documentation
│   ├── ROADMAP.md                # Architectural roadmap & improvement proposals
│   ├── TRAINING_REPORT.md        # Training report, loss curves & overfitting analysis
│   └── AGENTS.md                 # Agent guidelines
├── scripts/                      # Automation, CLI launchers & GPU profiling scripts
│   ├── train.py                  # Model training CLI launcher
│   ├── infer.py                  # Text generation inference CLI launcher
│   ├── evaluate.py               # Performance evaluation CLI launcher
│   └── profile.sh                # Nsight Systems / Compute profiling script
├── checkpoints/                  # Best model checkpoint & tokenizer artifact (Git-ignored)
├── data/                         # Training text corpora (Git-ignored)
├── config.yaml                   # Local model and training hyperparameter configuration
├── config.yaml.template          # Configuration template (tracked in Git)
├── pyproject.toml                # Package metadata and build config (pip install -e .)
├── README.md                     # Main repository introduction and usage guide
├── AGENTS.md                     # Root AI agent rules and guidelines
└── .gitignore                    # Git ignore rules
```

### Structural Rules
- **Separation of Concerns**: Write code only in its designated directory/layer (`src/`, `tests/`, `docs/`, `scripts/`).
- **Dependency Hierarchy**: Core lower modules (`src/config.py`, `src/model.py`, `src/dataset.py`) must never reverse-import CLI launchers (`scripts/`).
- **Adding New Files**: Place files according to their role and document them in `README.md` and `AGENTS.md`.

---

## Architectural Boundaries

1. **`src/config.py`**:
   - Manages dataclasses (`ModelConfig`, `TrainConfig`) and YAML serialization/deserialization.
   - Does not perform tensor calculations or execute training loops.
2. **`src/model.py`**:
   - Defines pure PyTorch neural network modules (`MiniLLM`, `TransformerBlock`, `CausalSelfAttention`, `FeedForward`).
   - Does not load dataset files or save checkpoints.
   - Maintains consistent `torch.profiler.record_function` markers across major computation paths.
3. **`src/dataset.py`**:
   - Handles tokenization (`CharTokenizer`) and DataLoader construction (`create_dataloaders`).
   - Does not compute forward passes or loss functions.
4. **`src/train.py` / `src/infer.py` / `src/evaluate.py` / `scripts/`**:
   - Composes config, model, and dataset modules to run training, inference, and evaluation pipelines.
   - Interacts with model weights via public interfaces without modifying internal tensor structures directly.

---

## Security & Data Integrity

### Security Rules
- **Credential Protection**: Never commit API keys, personal access tokens, or credentials.
- **Safe Serialization**:
  - Exercise caution with `torch.load` when loading external checkpoint files.
  - Always use `yaml.safe_load` for parsing YAML files.
- **Path Validation**:
  - Normalize file paths (`data_dir`, `checkpoint_dir`) to prevent directory traversal (`../`) vulnerabilities.

---

## Coding Standards

### General Standards
- **Standard Style**: Maintain clean, standard Python (PEP 8) style.
- **Clear Naming**: Use intuitive names that clearly convey purpose for variables, functions, and classes.
- **Type Annotations**: Provide `typing` annotations (`Tuple`, `List`, `Dict`, `Optional`, etc.) for public functions and class fields.
- **Docstrings & Comments**: Add explanations and tensor shape comments for complex tensor transformations (`transpose`, `view`, `chunk`, etc.).
- **Profiler Consistency**: Preserve hierarchical `torch.profiler.record_function` markers across model forward passes, backward passes, H2D transfers, and inference loops.

---

## Testing & Evaluation Standards

- **Feature Verification**:
  - Write standalone unit tests when adding new architectural modules (e.g., RoPE, RMSNorm, KV Cache) or tokenizer modifications.
- **Metric Consistency**:
  - Run [`evaluate.py`](file:///Users/chywoo/MyProjects/minillm/scripts/evaluate.py) after modifications to verify Cross-Entropy Loss, Perplexity (PPL), BLEU-4, and ROUGE-L calculations.
- **Edge Case Handling**:
  - Validate edge conditions such as sequence length overflow (`T > seq_len`), unknown character tokens, and empty string inputs.
