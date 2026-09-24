# gemmx-cpp

I'm building matrix multiplication in C++ from the naive version up, one optimization at a time, and measuring what each step actually does on my laptop.

Matrix multiplication (GEMM) is the core operation behind neural networks and most scientific computing. The math never changes, but *how* you write the loops can make the same computation run tens of times faster or slower, depending on how it uses the CPU's caches, SIMD units, and cores. I wanted to understand that gap properly, with measurements instead of guesses.

This is also my Research Methodology project, so every result comes with the raw data and a script to reproduce it.

## Progress

- [x] Naive reference kernel + correctness tests
- [x] Benchmark harness (raw CSV output + machine/compiler metadata)
- [x] EXP01: cache conflict at power-of-two sizes
- [ ] All six loop orders
- [ ] Cache blocking
- [ ] AVX2/FMA SIMD
- [ ] Register-blocked microkernel
- [ ] Multithreading
- [ ] Comparison with OpenBLAS
- [ ] P-core vs E-core experiments

## First result: the 512 cliff

The naive kernel runs at about 4.5 GFLOP/s for most sizes, then collapses at exactly one size:

| Size | 511 | **512** | 513 |
|---|---|---|---|
| double, GFLOP/s | 4.61 | **0.92** | 4.54 |

That's a 5× slowdown from adding one row. The same thing happens for float, but at 1024.

The common factor is that both cases make each row of the matrix exactly **4096 bytes** long. When the kernel walks down a column, every element then maps to the same small slice of the L1 cache, and the elements keep evicting each other. Sizes like 511 or 513 spread out across the cache and run fine.

Raw data: [`results/raw/exp01_pow2_conflict/`](results/raw/exp01_pow2_conflict/) · Notes: [`research/lab_notebook.md`](research/lab_notebook.md)

## Running it

You need Linux (I use WSL2), GCC 13 or newer, and CMake 3.24 or newer.

```bash
git clone https://github.com/Pratham-2105/gemmx-cpp.git
cd gemmx-cpp

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run a quick benchmark:

```bash
./build/gemmx_benchmarks --sizes 128,256,512 --reps 5 --out results/scratch
```

Reproduce an experiment exactly as I ran it:

```bash
./experiments/exp01_pow2_conflict.sh
```

Your numbers will differ from mine depending on your CPU, but the drop at 512 should show up on most x86 machines.

## How it's organized

```
include/gemmx/   matrix type and kernel interface
src/             the kernels
tests/           correctness tests
benchmarks/      benchmark tool
experiments/     one script per experiment
results/raw/     raw data from every experiment
research/        lab notebook: predictions before, results after
```

Every kernel has to pass the correctness tests before it gets benchmarked. Every result in this README comes from a file in `results/raw/`.

## Setup

Intel Core i5-13450HX (6 performance + 4 efficiency cores), 24 GB RAM, Windows 11 + WSL2, GCC 15.2. Benchmarks run plugged in, in Best performance mode.

## License

MIT
