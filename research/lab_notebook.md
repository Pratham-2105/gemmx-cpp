# GEMM-X Lab Notebook

Rules: predictions are committed BEFORE running. Outcomes cite run IDs in
results/raw/. Observation and interpretation are kept separate.
Scratch runs (results/scratch/) are never cited.

---

## EXP01 — Power-of-two stride conflict (reference kernel) {#exp01}

- **Date opened:** 2026-09-25
- **Question:** Does the i-j-k reference kernel slow down sharply at matrix
  sizes whose row stride is exactly 4 KB?
- **Hypothesis:** A 4 KB stride maps successive elements of a B column into
  few cache sets, causing conflict misses (Lam, Rothberg & Wolf, 1991).
- **Prediction:** double n=512 and float n=1024 show markedly lower GFLOP/s
  than their neighbours (e.g. 511/513, 1023/1025); non-power-of-two sizes
  show no comparable dip.
- **Preliminary (scratch, uncited):** double 511 → 3.01, 512 → 0.63,
  513 → 3.09 GFLOP/s.
- **Command:** `experiments/exp01_pow2_conflict.sh`
- **Conditions:** laptop plugged in; Windows power mode: ______;
  other heavy apps closed: yes/no
- **Runs:** _(fill in run IDs after running)_
- **Observation:** _(what the numbers show, no explanation)_
- **Interpretation:** _(why, with caveats: TLB effects cannot be separated
  without hardware counters, unavailable in WSL2)_
- **Status:** open
