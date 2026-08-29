#!/usr/bin/env bash
# ==============================================================================
# MILM GPU Profiling Automation Script (Nsight Systems & Nsight Compute)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODE="${1:-nsys}"

if [ "$MODE" == "nsys" ]; then
    echo "Running NVIDIA Nsight Systems profiling..."
    nsys profile -t cuda,nvtx,osrt -s cpu --force-overwrite=true --output=llm_profile python scripts/train.py
    echo "Profiling complete: generated llm_profile.nsys-rep"

elif [ "$MODE" == "ncu" ]; then
    echo "Running NVIDIA Nsight Compute kernel analysis (FlashAttention_SDPA)..."
    ncu --nvtx --nvtx-include "FlashAttention_SDPA" --set full --force-overwrite -o profile_sdpa_kernel python scripts/train.py
    echo "Kernel analysis complete: generated profile_sdpa_kernel.ncu-rep"

else
    echo "Usage: ./scripts/profile.sh [nsys|ncu]"
    exit 1
fi
