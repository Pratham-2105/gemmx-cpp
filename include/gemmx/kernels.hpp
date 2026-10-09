#pragma once

#include <string_view>
#include <vector>

#include "gemmx/gemm.hpp"

namespace gemmx {

// Every kernel has exactly this signature (the contract in gemm.hpp).
template <typename T>
using KernelFn = void (*)(const Matrix<T> &, const Matrix<T> &, Matrix<T> &);

template <typename T> struct NamedKernel {
  std::string_view name;
  KernelFn<T> fn;
  int block_size; // 0 = not a blocked kernel
};

// THE registry. Add one line per new kernel; tests and benchmarks pick it up.
template <typename T> std::vector<NamedKernel<T>> all_kernels() {
  return {
      {"reference", &gemm_reference<T>, 0}, // i-j-k
      {"loop_ikj", &gemm_loop_ikj<T>, 0},   {"loop_jik", &gemm_loop_jik<T>, 0},
      {"loop_jki", &gemm_loop_jki<T>, 0},   {"loop_kij", &gemm_loop_kij<T>, 0},
      {"loop_kji", &gemm_loop_kji<T>, 0},
  };
}

} // namespace gemmx
