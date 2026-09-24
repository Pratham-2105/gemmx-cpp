#!/usr/bin/env bash
# EXP01 — Power-of-two stride conflict in the reference (i-j-k) kernel.
# Hypothesis: when the row stride of B is exactly 4 KB (double n=512,
# float n=1024), the column walk maps into few cache sets, causing conflict
# misses and a sharp throughput drop relative to neighbouring sizes.
# Lab notebook: research/lab_notebook.md#exp01
source "$(dirname "$0")/common.sh"

run_experiment exp01_pow2_conflict --kernels reference --dtypes double \
  --sizes 496,500,504,508,511,512,513,516,520,528 --reps 7

run_experiment exp01_pow2_conflict --kernels reference --dtypes float \
  --sizes 1000,1016,1023,1024,1025,1032,1040 --reps 7
