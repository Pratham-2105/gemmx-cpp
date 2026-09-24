# GEMM-X Lab Notebook

Standard conditions for all formal runs: laptop plugged in, Windows power
mode "Best performance", heavy apps closed. Deviations are noted per entry.
Predictions are written before running. Scratch runs are never cited.

---

## EXP01 — Power-of-two stride conflict (reference kernel)

**Prediction:** double n=512 and float n=1024 (B row stride exactly 4 KB)
will be far slower than neighbouring sizes, because the column walk maps
into few cache sets (conflict misses; Lam, Rothberg & Wolf, 1991).

**Result: held.** Standard conditions, median of 7
(runs 20260924-212250 double, 20260924-212257 float):
- double 512 → 0.920 GFLOP/s vs 4.613 (511) / 4.537 (513): ~5.0x drop
- float 1024 → 0.788 vs 4.175 (1023) / 4.315 (1025): ~5.4x drop
- smaller dips at double 496 (3.258) and 528 (3.270)
  vs 504 (4.141) and 520 (3.944)

Both collapses share a 4096-byte row stride, not a matrix size or data
type. Interpretation: with 64-B-aligned rows and a stride of s cache lines,
a column spans 64/gcd(s,64) L1 sets (P-core L1D: 48 KB, 12-way, 64 sets).
s=64 -> 1 set (total conflict); s=62/66 (n=496/528) -> 32 sets (partial
conflict); s=63/65 (n=504/520) -> 64 sets (fits). Not confirmed by hardware
counters (unavailable in WSL2); TLB effects and P-/E-core placement unknown.

**Robustness:** an earlier run in power-efficiency mode (deviation; runs
20260924-211248, 20260924-211259) was ~1.45x slower overall but showed the
same pattern (double 512 -> 0.672 vs 3.185/2.983; float 1024 -> 0.557 vs
3.076/2.321). The effect does not depend on power mode, but absolute
throughput does, by ~1.45x: power mode is a major confounder.
