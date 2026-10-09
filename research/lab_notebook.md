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

---

## EXP02 — Loop order (all six), portable vs native

**Prediction:** ikj and kij will be fastest and nearly equal (inner loop
streams rows of B and C); jki and kji slowest (inner loop walks columns of
A and C); ijk (reference) and jik in between. The best-to-worst gap grows
with n. In the native build, ikj/kij gain substantially over portable
(auto-vectorized with wider AVX2 vectors), while ijk/jik barely change (their
inner loop is a reduction, which the compiler may not vectorize without
-ffast-math). The 4 KB-stride cliffs (float 1024, double 512/1024) appear in
the column-walking orders (ijk, jik, jki, kji) but not in ikj/kij.

**Result: mostly held; one partial miss.** Runs 20261009-062519 (portable),
20261009-063213 (native), median of 5. A first attempt was discarded: a stale
benchmark binary from older code was run, so its metadata did not match the
code (fixed by always building fresh).
- Ranking held: ikj/kij fastest, jki/kji slowest (below reference), ijk/jik
  between. Native float best/worst gap grows from ~29x (n=128) to ~85x
  (n=1024).
- Native build sped up ikj by 1.5–2.7x (float) and 1.3–2.5x (double);
  reference and jik unchanged (reduction not vectorized). The native gain
  shrinks at large n (float ikj 2.7x at 256, 1.5x at 1024), consistent with
  the kernel becoming memory-bound once B exceeds L2.
- Power-of-two cliffs appear only in column-walking orders (reference float
  1024: 0.51 vs 2.25 at 768); ikj shows none (22.7 vs 22.1).
- Miss: ikj and kij are not equal at large n (native float 1024: 22.7 vs
  13.9). kij streams the whole of C once per k; ikj keeps one C row in cache
  across all k.
Next: blocking (EXP03) to recover the memory-bound loss.

---

## EXP03 — Cache-blocking block-size sweep

**Prediction:** GFLOP/s vs block size has an interior optimum, expected in
the 64–256 range: BS=16 may be no faster (or slower) than unblocked loop_ikj
because of short inner loops, while the best block size clearly beats
loop_ikj at n=2000/2048, where B no longer fits in L2. The optimum for
double is at or below the optimum for float (twice the bytes per tile).
Power-of-two sizes (1024, 2048) may still underperform 1000/2000 under
blocking, due to self-interference between tile rows 4 KB apart (Lam,
Rothberg & Wolf, 1991).

**Result: held.** Run 20261009-063956 (native), median of 5.
- Interior optimum at BS=64–96 for both types; BS=16 is slower than
  unblocked ikj at n≈1000 (float 15.5 vs 23.0).
- Best block vs unblocked ikj: float 1.5x (1000) to 2.2x (2000); double 1.7x
  (1000) to 4.0x (2000, where unblocked ikj collapses to 4.5 GFLOP/s).
- Power-of-two sizes remain slower under blocking: best block at 2048 is 27%
  (float) and 32% (double) below 2000; large blocks degrade most (float b256:
  30.9 at 2000 vs 17.8 at 2048), consistent with tile self-interference
  (Lam et al., 1991).
- Double optimum (64) at or below float (64–96), weakly.
- Unexplained: double b96 dips at 1000/2000 (single-call samples; possibly
  noise).
Next: explicit AVX2/FMA (V3).
