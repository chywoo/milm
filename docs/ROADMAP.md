# MILM Roadmap & Architectural Recommendations

This document outlines the phased improvements, architectural upgrades, and inference/training optimizations planned for the **MILM** project.

---

## Roadmap Overview

| Priority | Feature / Proposal | Category | Complexity | Expected Impact |
| :---: | :--- | :---: | :---: | :--- |
| **P0** | **Subword Tokenizer (BPE / tiktoken)** | Data / Tokenizer | Medium | 3~5x more word information per sequence window; fixes token fragmentation |
| **P0** | **KV Cache (Key-Value Caching)** | Inference Optimization | Medium | Reduces generation time complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ |
| **P1** | **Rotary Position Embedding (RoPE)** | Architecture | Medium | Context length extrapolation and improved relative positional representation |
| **P1** | **Dataset Chunking & Non-overlapping Sequences** | Data Pipeline | Low | Eliminates 99% sliding-window redundancy; speeds up epoch throughput |
| **P2** | **RMSNorm & SwiGLU Activation** | Architecture | Low | Aligns with modern LLM architectures (LLaMA/Mistral); lowers compute overhead |
| **P2** | **FlashAttention-2 / Triton Kernel Tuning** | GPU / Kernels | Medium | Maximizes memory bandwidth efficiency with NVTX trace integration |

---

## Detailed Improvement Proposals

### 1. Subword Tokenizer (BPE / WordPiece)
- **Current State**: The existing `CharTokenizer` operates on raw character-level vocabulary (~105 tokens). With `seq_len=128`, a single context window holds only ~20–30 words.
- **Proposal**:
  - Implement **Byte-Level Byte-Pair Encoding (BPE)** using Hugging Face `tokenizers` or OpenAI's `tiktoken`.
  - Configure vocabulary size (`vocab_size`) between 4,096 and 16,384.
- **Expected Impact**:
  - Encodes 4–5x longer context for the same sequence length (`seq_len`).
  - Strengthens semantic word representations and accelerates training convergence.

---

### 2. KV Cache (Key-Value Caching) Inference Engine
- **Current State**: In `src/infer.py`, generating each new token re-passes the entire sequence `input_ids[:, -seq_len:]` through the model, recomputing $Q, K, V$ for all prior tokens ($\mathcal{O}(N^2)$).
- **Proposal**:
  - Introduce a `past_key_values` KV Cache buffer in `CausalSelfAttention`.
  - During autoregressive generation, compute $Q$ only for the newest token ($T=1$), concatenate new $K, V$ to existing caches, and perform $\mathcal{O}(1)$ attention per step.
- **Expected Impact**:
  - 5–10x latency reduction when generating 100–200 tokens.

---

### 3. Rotary Position Embedding (RoPE)
- **Current State**: Learned absolute positional embeddings via `nn.Embedding(seq_len, d_model)`.
- **Proposal**:
  - Adopt **RoPE (Rotary Position Embedding)** as used in modern SOTA architectures (LLaMA, Mistral, Gemma).
  - Apply 2D rotation matrices to Query ($Q$) and Key ($K$) vectors to naturally encode relative token distances.
- **Expected Impact**:
  - Enables length extrapolation to inference sequences longer than the training context window.

---

### 4. Dataset Chunking & Packed Sequences
- **Current State**: `TextDataset` creates samples via single-character sliding windows, producing $N - \text{seq\_len}$ samples with up to 99% duplication between adjacent samples.
- **Proposal**:
  - Implement non-overlapping sequence chunking (`seq_len` chunks) or a packed sequence pipeline separated by `<|endoftext|>` delimiters.
- **Expected Impact**:
  - Drastically reduces step count per epoch and improves data diversity to mitigate overfitting.

---

### 5. Modern Architecture Upgrades (RMSNorm & SwiGLU)
- **Current State**: Standard LayerNorm (`nn.LayerNorm`) and standard FFN (Linear + GELU + Linear).
- **Proposal**:
  - **RMSNorm (Root Mean Square Layer Normalization)**: Normalizes using RMS without mean calculation, reducing memory operations.
  - **SwiGLU (Swish Gated Linear Unit)**: Replaces standard FFN with $FFN(x) = (\text{Swish}(xW) \otimes xV)W_2$ for higher representational capacity per parameter.

---

## Long-Term Goals

1. **Distributed Training (DDP / FSDP)**: Multi-GPU and distributed training support.
2. **SFT / DPO Fine-tuning Pipeline**: Instruction tuning and direct preference optimization (DPO/RLHF).
3. **ONNX / TensorRT-LLM Export**: Optimized inference runtimes for edge and production deployment.
