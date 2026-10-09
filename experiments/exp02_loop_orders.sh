#!/usr/bin/env bash
# EXP02 — Effect of loop order (all six), portable vs -march=native.
# Lab notebook: research/lab_notebook.md (EXP02)
source "$(dirname "$0")/common.sh"

# Kernels are listed explicitly so later kernels never sneak into a rerun.
KERNELS=reference,loop_ikj,loop_jik,loop_jki,loop_kij,loop_kji
SIZES=128,256,512,768,1024

run_experiment exp02_loop_orders --kernels "$KERNELS" --dtypes float,double --sizes "$SIZES" --reps 5
run_experiment_native exp02_loop_orders --kernels "$KERNELS" --dtypes float,double --sizes "$SIZES" --reps 5
