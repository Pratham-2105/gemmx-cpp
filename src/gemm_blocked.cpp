#include <algorithm>

#include "gemmx/gemm.hpp"

namespace gemmx {

// Cache-blocked GEMM: i-k-j inside each tile, tiles visited in ii-kk-jj order.
//
// Reuse inside one (ii, kk, jj) tile step:
//   - the BS x BS tile of B is reused for every row i of the tile (BS times),
//   - each a(i,k) is reused across the BS columns j,
//   - each row segment of C stays hot across the BS values of k.
// The tile loops use std::min so sizes that are not a multiple of BS work.
template <typename T, std::size_t BS>
void gemm_blocked(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) {
  static_assert(BS > 0, "block size must be positive");
  check_gemm_args(A, B, C);

  const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
  const T *a = A.data();
  const T *b = B.data();
  C.fill(T{0});
  T *c = C.data();

  for (std::size_t ii = 0; ii < M; ii += BS) {
    const std::size_t i_end = std::min(ii + BS, M);
    for (std::size_t kk = 0; kk < K; kk += BS) {
      const std::size_t k_end = std::min(kk + BS, K);
      for (std::size_t jj = 0; jj < N; jj += BS) {
        const std::size_t j_end = std::min(jj + BS, N);

        // One tile step: C[ii:i_end, jj:j_end] += A[ii:i_end, kk:k_end] *
        // B[kk:k_end, jj:j_end]
        for (std::size_t i = ii; i < i_end; ++i) {
          T *crow = c + i * N;
          for (std::size_t k = kk; k < k_end; ++k) {
            const T aik = a[i * K + k];
            const T *brow = b + k * N;
            for (std::size_t j = jj; j < j_end; ++j) {
              crow[j] += aik * brow[j];
            }
          }
        }
      }
    }
  }
}

#define GEMMX_INSTANTIATE_BLOCKED(BS)                                          \
  template void gemm_blocked<float, BS>(                                       \
      const Matrix<float> &, const Matrix<float> &, Matrix<float> &);          \
  template void gemm_blocked<double, BS>(                                      \
      const Matrix<double> &, const Matrix<double> &, Matrix<double> &);

GEMMX_INSTANTIATE_BLOCKED(16)
GEMMX_INSTANTIATE_BLOCKED(32)
GEMMX_INSTANTIATE_BLOCKED(48)
GEMMX_INSTANTIATE_BLOCKED(64)
GEMMX_INSTANTIATE_BLOCKED(96)
GEMMX_INSTANTIATE_BLOCKED(128)
GEMMX_INSTANTIATE_BLOCKED(192)
GEMMX_INSTANTIATE_BLOCKED(256)

#undef GEMMX_INSTANTIATE_BLOCKED

} // namespace gemmx
