#!/usr/bin/env bash
# EXP05 — Packed panels + register-blocked microkernel (V4) vs V2/V3.
# Native build only: compares against the strongest compiler baseline.
# Lab notebook: research/lab_notebook.md (EXP05)
source "$(dirname "$0")/common.sh"

KERNELS=blocked_b64,avx2_b64,packed
SIZES=512,1000,1024,2000,2048

run_experiment_native exp05_packed --kernels "$KERNELS" --dtypes float,double --sizes "$SIZES" --reps 5
