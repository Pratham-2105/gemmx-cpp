# gemmx-cpp

An experimental study of dense matrix multiplication (GEMM) optimization in C++20 on a hybrid P-core/E-core x86 laptop CPU.

The goal is not a new algorithm or a BLAS replacement. It is a **reproducible, measured account** of how loop ordering, cache blocking, SIMD vectorization, register blocking, and multithreading change GEMM performance, and *why*, on one clearly documented machine. Every kernel is checked against an error-bound correctness suite, and every reported number traces back to a committed raw-data file.

> **Status:** early. Reference kernel, correctness suite, benchmark harness, and the first experiment (EXP01) are complete. The optimization ladder is in progress.

---

## Findings so far

### EXP01 — 4 KB stride cliff in the naive kernel

The textbook i-j-k kernel walks down columns of B. When a row of B is exactly **4096 bytes** long, throughput collapses:

| dtype | n−1 | **n (4 KB stride)** | n+1 | drop |
|---|---|---|---|---|
| double | 511 → 4.61 GFLOP/s | **512 → 0.92** | 513 → 4.54 | ~5.0× |
| float | 1023 → 4.18 GFLOP/s | **1024 → 0.79** | 1025 → 4.32 | ~5.4× |

The two cliffs occur at different matrix sizes and data types, but at the same **byte stride**. This is consistent with cache-set conflict misses: with 64-byte-aligned rows and a stride of *s* cache lines, a column walk maps into only 64/gcd(*s*, 64) of the L1 cache's 64 sets. Smaller dips at double n = 496 and 528 (stride of 62 and 66 lines, so 32 sets) fit the same model.

Median of 7 repetitions; raw data in [`results/raw/exp01_pow2_conflict/`](results/raw/exp01_pow2_conflict/); prediction and interpretation in [`research/lab_notebook.md`](research/lab_notebook.md). This is a model-based interpretation; hardware-counter confirmation is planned.

---

## Optimization ladder

| Stage | Kernel | Status |
|---|---|---|
| V0 | Reference i-j-k | ✅ done |
| V1 | All six loop orders (portable vs `-march=native`) | 🔜 next |
| V2 | Cache blocking, block-size sweep | planned |
| V3 | Explicit AVX2/FMA intrinsics | planned |
| V4 | Register-blocked microkernel + packing | planned |
| V5 | OpenMP multithreading | planned |
| — | OpenBLAS baseline | planned |
| — | P-core vs E-core scheduling study | planned |

---

## Correctness

No kernel is benchmarked before it passes the test suite.

- **Error-bound tolerance, not arbitrary epsilons.** Each output element is checked against a double-precision oracle using the standard dot-product rounding bound, |computed − exact| ≤ 2·K·ε·(|A||B|)ᵢⱼ. The bound holds for any summation order, so the same test stays valid for blocked, SIMD, and multithreaded kernels.
- **Awkward shapes.** 1-sized dimensions, primes, and sizes one past a power of two, for both `float` and `double`.
- **NaN pre-fill.** Output matrices start as NaN, so a kernel that skips any element fails.
- **Verification before timing.** The benchmark re-checks every configuration with a randomized O(n²) product check (Freivalds) before timing it, and never times a configuration that fails.

---

## Benchmark methodology

- Setup (allocation, data generation) is outside the timed region.
- One warm-up call per configuration, also used for calibration and verification.
- Calls are repeated until each timed sample lasts ≥ 50 ms; the iteration count is recorded.
- `std::chrono::steady_clock` (monotonic).
- **Every repetition is written as a raw CSV row;** summaries are computed later from raw data.
- Each run writes a JSON metadata sidecar: CPU model, OS (including WSL detection), compiler and version, build type, compiler flags, `-march=native` on/off, git commit, seed, sizes, and repetitions.
- Input data is deterministic and identical across compilers (the random generator is converted manually, since `std::uniform_real_distribution` differs between standard libraries).

**Standard conditions for formal runs:** laptop on AC power (enforced by the experiment script), Windows power mode *Best performance*, heavy applications closed. Power mode alone changed absolute throughput by ~1.45× in EXP01, so it is held fixed.

---

## Build and run

**Requirements:** x86-64 Linux (tested on WSL2, Ubuntu), GCC ≥ 13 (C++20 `<format>`), CMake ≥ 3.24, internet access on first configure (GoogleTest is fetched automatically).

```bash
git clone https://github.com/Pratham-2105/gemmx-cpp.git
cd gemmx-cpp

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

**Ad-hoc benchmark** (output goes to the gitignored scratch directory):

```bash
./build/gemmx_benchmarks --sizes 128,256,512 --reps 5 --out results/scratch
./build/gemmx_benchmarks --help
```

**Reproduce an experiment.** This builds a fresh Release binary, runs the tests, and writes raw results tagged with the current git commit:

```bash
./experiments/exp01_pow2_conflict.sh
```

Absolute numbers depend on your CPU; the *shape* of the result (the cliff at a 4 KB stride) should reproduce on any x86 CPU with a 64-set L1 data cache.

**Build options:**

| Option | Default | Effect |
|---|---|---|
| `GEMMX_NATIVE` | OFF | Compile with `-march=native`. Kept off by default because whether the compiler may use AVX2 on scalar kernels is an experimental variable. |
| `GEMMX_BUILD_TESTS` | ON | Build the GoogleTest suite |
| `GEMMX_BUILD_BENCHMARKS` | ON | Build `gemmx_benchmarks` |

---

## Repository layout

```
include/gemmx/        Matrix type, kernel contract, kernel registry
src/                  Kernel implementations
tests/                GoogleTest correctness suite
benchmarks/           Benchmark harness + system-info collection
experiments/          One script per experiment (fixed arguments)
results/raw/          Raw CSV + JSON from formal runs (committed, never edited)
research/             Lab notebook: predictions (before) and results (after)
```

**Evidence policy:** formal runs are written only by the experiment scripts, committed, and never edited or deleted. Each run's metadata records the exact code commit (marked `-dirty` if the code had uncommitted changes). Predictions are committed before experiments run.

---

## Test machine

Intel Core i5-13450HX (6 P-cores + 4 E-cores, 16 logical CPUs), 24 GB RAM, Windows 11 + WSL2, GCC 15.2.

**Known limitations:** single machine; WSL2 exposes no hardware performance counters and no reliable P-core/E-core thread placement, so cache-effect interpretations are model-based for now.

---

## References

- K. Goto, R. A. van de Geijn. *Anatomy of High-Performance Matrix Multiplication.* ACM TOMS 34(3), 2008.
- M. S. Lam, E. E. Rothberg, M. E. Wolf. *The Cache Performance and Optimizations of Blocked Algorithms.* ASPLOS 1991.
- T. M. Low, F. D. Igual, T. M. Smith, E. S. Quintana-Ortí. *Analytical Modeling Is Enough for High-Performance BLIS.* ACM TOMS 43(2), 2016.
- T. Hoefler, R. Belli. *Scientific Benchmarking of Parallel Computing Systems.* SC15.
