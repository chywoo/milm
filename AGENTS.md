# AGENTS.md

## Purpose
The project name is **MILM** (Lightweight Transformer Decoder-Only Language Model), built for LLM study, experimentation, and research.

This document defines workflows, operational boundaries, coding practices, and guidelines for AI agents and developers working on the MILM repository.
Follow these instructions during code analysis, file modification, environment checks, testing, and pull request preparation.

If specific modules or documentation (e.g., `README.md`, `ROADMAP.md`) state more specific rules, those specialized documents take precedence.

---

## Development Guidelines

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Tools Execution
### GIT
- Do not request a Git commit.
- Do not perform Git operations such as “commit,” “update,” “push,” “pull,” “restore,” “reset,” or “restore” without permission. 
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
