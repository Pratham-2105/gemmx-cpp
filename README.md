# gemmx-cpp

I'm building matrix multiplication in C++ from the naive version up, one optimization at a time, and measuring what each step actually does on my laptop.

Matrix multiplication (GEMM) is the core operation behind neural networks and most scientific computing. The math never changes, but *how* you write the loops can make the same computation run tens of times faster or slower, depending on how it uses the CPU's caches, SIMD units, and cores. I wanted to understand that gap properly, with measurements instead of guesses.

This is also my Research Methodology project, so every result comes with the raw data and a script to reproduce it.

## Progress

- [x] Naive reference kernel + correctness tests
- [x] Benchmark harness (raw CSV output + machine/compiler metadata)
- [x] EXP01: Cache conflict at power-of-two sizes
- [x] EXP02: All six loop orders
- [x] EXP03: Cache blocking
- [x] EXP04: AVX2/FMA SIMD
- [x] EXP05: Register-blocked microkernel + packing
- [ ] Multithreading
- [ ] Comparison with OpenBLAS
- [ ] P-core vs. E-core experiments

## Results so far

All numbers are single-threaded, median of 5–7 runs, in GFLOP/s. Kernels are only compared within the same run.

### 1. The 512 cliff

The naive kernel runs at about 4.5 GFLOP/s for most sizes, then collapses at exactly one size:

| Size | 511 | **512** | 513 |
|---|---:|---:|---:|
| double | 4.61 | **0.92** | 4.54 |

That's a 5× slowdown from adding one row. The same thing happens for `float`, but at 1024.

The common factor is that both cases make each row of the matrix exactly **4096 bytes** long. When the kernel walks down a column, every element then maps to the same small slice of the L1 cache, and the elements keep evicting each other. Sizes like 511 or 513 spread out across the cache and run fine.

### 2. Loop order alone: up to 85× apart

There are six ways to nest the three loops of a matrix multiply. They all do exactly the same arithmetic:

| Order | n = 256 | n = 1024 |
|---|---:|---:|
| `ikj` | **41.2** | **22.7** |
| `kij` | 25.8 | 13.9 |
| `ijk` (naive) | 2.9 | 0.53 |
| `jki` | 0.71 | 0.27 |

*Float, compiled with `-march=native`.*

The fast orders walk along rows of memory, so the CPU streams data and the compiler can use SIMD. The slow ones walk down columns and jump 4 KB on every step. Turning on AVX2 (`-march=native`) made `ikj` up to 2.7× faster but did nothing for the naive order: its inner loop is a running sum, and the compiler isn't allowed to split that across SIMD lanes.

### 3. Cache blocking: up to 4× on large matrices

Once the matrix stops fitting in cache, even the best loop order slows down, because it re-reads all of B from main memory for every row. Blocking works on small tiles that stay in cache instead:

| Kernel (double) | n = 1000 | n = 2000 |
|---|---:|---:|
| `ikj`, no blocking | 10.9 | 4.5 |
| Blocked, 64×64 tiles | **18.6** | **18.1** |

The best tile size was 64–96. Tiles that are too small waste time on loop overhead; tiles that are too big stop fitting in cache. Power-of-two sizes still lose about 30% even with blocking (2048 vs. 2000), for the same reason as the 512 cliff.

### 4. Hand-written AVX2: the compiler already did it

Next I rewrote the blocked kernel's inner loop with AVX2/FMA intrinsics, processing 8 floats (or 4 doubles) per instruction. The AVX2 code is switched on per function and only used if the CPU supports it, so the rest of the program stays portable.

| Kernel (float, n = 1000) | Portable build | `-march=native` build |
|---|---:|---:|
| Blocked, compiler code | 13.5 | **32.3** |
| Blocked, my AVX2 | **27.6** | 26.1 |

Against the portable build (old SSE2 only), my version was 1.5–2× faster. But with `-march=native`, the compiler vectorizes the same loop by itself, and it beat my version by up to ~19%. Writing intrinsics alone wasn't the win. Both versions still load and store C on every single multiply-add, and that memory traffic is the real limit.

### 5. Microkernel + packing: ~3× faster, and the 4 KB cliff is gone

This is the approach real BLAS libraries use (Goto/BLIS). Two ideas:

- **Keep C in registers.** A small "microkernel" computes a 6×16 tile of C (6×8 for double) in 12 AVX2 registers for 256 steps, and only then writes it back to memory.
- **Pack A and B first.** Copy blocks of A and B into buffers laid out in exactly the order the microkernel reads them, so it streams memory in a straight line. Block sizes come from my CPU's cache sizes (`docs/cpu_caches.txt`).

| Kernel (float) | n = 1000 | n = 2000 | n = 2048 |
|---|---:|---:|---:|
| Blocked, compiler code | 46.3 | 42.9 | 30.3 |
| **Packed microkernel** | **125.0** | **106.3** | **98.1** |

That's 2.5–3.5× faster for float and 1.9–2.9× for double, up to about 44% of the theoretical peak of one core. Because the packed buffers are contiguous, the power-of-two penalty mostly disappears: 2048 is only ~8% slower than 2000 now, instead of ~30%.

Float performance drops at n = 2000. My best guess is that the packed copy of B stops fitting in L2 cache at that size, and double drops at exactly the size where its (twice as big) copy stops fitting. I haven't proven it yet; a block-size sweep will.

**Data and notes:** [Raw benchmark data](results/raw/) · [Research lab notebook](research/lab_notebook.md)

## Running it

You need Linux (I use WSL2), GCC 13 or newer, and CMake 3.24 or newer. The AVX2 and packed kernels need an x86 CPU with AVX2 and FMA (most Intel/AMD CPUs since ~2013); on other CPUs they're skipped automatically.

```bash
git clone https://github.com/Pratham-2105/gemmx-cpp.git
cd gemmx-cpp

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Run a quick benchmark

```bash
./build/gemmx_benchmarks --sizes 128,256,512 --reps 5 --out results/scratch
./build/gemmx_benchmarks --help
```

### Reproduce an experiment

```bash
./experiments/exp01_pow2_conflict.sh
./experiments/exp02_loop_orders.sh
./experiments/exp03_block_size.sh
./experiments/exp04_avx2.sh
./experiments/exp05_packed.sh
```

Each script builds the code fresh, runs the tests, and only then benchmarks. It also refuses to run on battery, because power mode alone changed my results by ~1.45×. EXP02 and EXP03 take 10–20 minutes each, because the slow kernels really are slow. Your numbers will differ depending on your CPU, but the patterns should show up on most x86 machines.

## How it's organized

```text
include/gemmx/   matrix type, kernel interface, CPU feature detection
src/             the kernels
tests/           correctness tests
benchmarks/      benchmark tool
experiments/     one script per experiment
results/raw/     raw data from every experiment
research/        lab notebook: predictions before, results after
docs/            machine details (cache sizes)
```

Every kernel has to pass the correctness tests before it gets benchmarked, and every benchmark run checks the result again before timing it. Every result in this README comes from a file in `results/raw/`.

## Setup

Intel Core i5-13450HX (6 performance + 4 efficiency cores), 24 GB RAM, Windows 11 + WSL2, GCC 15.2. Benchmarks run plugged in, in Best performance mode.

## License

MIT
