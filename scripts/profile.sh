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
    echo "NVIDIA Nsight Systems 프로파일링 실행 중..."
    nsys profile -t cuda,nvtx,osrt -s cpu --force-overwrite=true --output=llm_profile python scripts/train.py
    echo "프로파일링 완료: llm_profile.nsys-rep 생성됨"

elif [ "$MODE" == "ncu" ]; then
    echo "NVIDIA Nsight Compute 커널 분석 실행 중 (FlashAttention_SDPA)..."
    ncu --nvtx --nvtx-include "FlashAttention_SDPA" --set full --force-overwrite -o profile_sdpa_kernel python scripts/train.py
    echo "커널 분석 완료: profile_sdpa_kernel.ncu-rep 생성됨"

else
    echo "사용법: ./scripts/profile.sh [nsys|ncu]"
    exit 1
fi
