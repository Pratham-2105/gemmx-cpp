#pragma once

#include <stdexcept>

#include "gemmx/matrix.hpp"

namespace gemmx {

// ---------------------------------------------------------------------------
// Contract shared by EVERY kernel in this project:
//   A is M x K, B is K x N, C is M x N, all row-major.
//   On return, C = A * B. C's previous contents are overwritten.
//   C must not be the same object as A or B.
//   Throws std::invalid_argument on a shape mismatch or aliasing.
//
// Kernels are templates, explicitly instantiated for float and double in
// their .cpp files.
// ---------------------------------------------------------------------------

template <typename T>
void check_gemm_args(const Matrix<T> &A, const Matrix<T> &B,
                     const Matrix<T> &C) {
  if (A.cols() != B.rows() || C.rows() != A.rows() || C.cols() != B.cols()) {
    throw std::invalid_argument("gemmx: incompatible matrix shapes");
  }
  if (&C == &A || &C == &B) {
    throw std::invalid_argument("gemmx: output C must not alias an input");
  }
}

// Version 0: textbook i-j-k triple loop.
// Serves as both the readable correctness reference and the performance
// baseline.
template <typename T>
void gemm_reference(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C);

} // namespace gemmx
