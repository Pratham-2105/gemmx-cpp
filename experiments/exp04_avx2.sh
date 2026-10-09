#!/usr/bin/env bash
# EXP04 — Explicit AVX2/FMA (V3) vs compiler vectorization.
# Portable build: avx2_b64 vs SSE2-only blocked_b64.
# Native build:   avx2_b64 vs auto-vectorized (AVX2) blocked_b64.
# Lab notebook: research/lab_notebook.md (EXP04)
source "$(dirname "$0")/common.sh"

KERNELS=loop_ikj,blocked_b64,avx2_b64
SIZES=512,1000,1024,2000,2048

run_experiment exp04_avx2_portable --kernels "$KERNELS" --dtypes float,double --sizes "$SIZES" --reps 5
run_experiment_native exp04_avx2_native --kernels "$KERNELS" --dtypes float,double --sizes "$SIZES" --reps 5
