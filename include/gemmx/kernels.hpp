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
};

// THE registry. Add one line per new kernel; tests and benchmarks pick it up.
template <typename T> std::vector<NamedKernel<T>> all_kernels() {
  return {
      {"reference", &gemm_reference<T>},
  };
}

} // namespace gemmx
