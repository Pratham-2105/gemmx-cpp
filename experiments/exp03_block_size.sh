#!/usr/bin/env bash
# EXP03 — Cache-blocking block-size sweep (native build), vs unblocked i-k-j.
# Lab notebook: research/lab_notebook.md (EXP03)
source "$(dirname "$0")/common.sh"

KERNELS=loop_ikj,blocked_b16,blocked_b32,blocked_b48,blocked_b64,blocked_b96,blocked_b128,blocked_b192,blocked_b256
SIZES=1000,1024,2000,2048

run_experiment_native exp03_block_size --kernels "$KERNELS" --dtypes float,double --sizes "$SIZES" --reps 5
