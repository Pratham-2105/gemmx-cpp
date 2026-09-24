#include "gemmx/gemm.hpp"

namespace gemmx {

template <typename T>
void gemm_reference(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) {
  check_gemm_args(A, B, C);

  const std::size_t M = A.rows();
  const std::size_t K = A.cols();
  const std::size_t N = B.cols();

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T sum = T{0};
      for (std::size_t k = 0; k < K; ++k) {
        // A(i, k) walks along a row: contiguous.
        // B(k, j) walks down a column: stride of N elements per step.
        sum += A(i, k) * B(k, j);
      }
      C(i, j) = sum;
    }
  }
}

template void gemm_reference<float>(const Matrix<float> &,
                                    const Matrix<float> &, Matrix<float> &);
template void gemm_reference<double>(const Matrix<double> &,
                                     const Matrix<double> &, Matrix<double> &);

} // namespace gemmx
