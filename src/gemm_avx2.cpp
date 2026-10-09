#include <algorithm>
#include <cstddef>
#include <immintrin.h>
#include <type_traits>

#include "gemmx/gemm.hpp"

namespace gemmx {
namespace {

constexpr std::size_t kBS = 64; // tile size (the best size from EXP03)

// crow[j..j_end) += aik * brow[j..j_end), using AVX2 + FMA.
// This is the old innermost loop of gemm_blocked, now vectorized.
template <typename T>
__attribute__((target("avx2,fma"), always_inline)) inline void
axpy_row(T *crow, const T *brow, T aik, std::size_t j, std::size_t j_end) {
  if constexpr (std::is_same_v<T, double>) {
    const __m256d va = _mm256_set1_pd(aik); // aik copied into all 4 lanes
    for (; j + 4 <= j_end; j += 4) {
      __m256d vc = _mm256_loadu_pd(crow + j);
      const __m256d vb = _mm256_loadu_pd(brow + j);
      vc = _mm256_fmadd_pd(va, vb, vc); // vc = va * vb + vc
      _mm256_storeu_pd(crow + j, vc);
    }
  } else {
    const __m256 va = _mm256_set1_ps(aik); // 8 lanes for float
    for (; j + 8 <= j_end; j += 8) {
      __m256 vc = _mm256_loadu_ps(crow + j);
      const __m256 vb = _mm256_loadu_ps(brow + j);
      vc = _mm256_fmadd_ps(va, vb, vc);
      _mm256_storeu_ps(crow + j, vc);
    }
  }
  // Leftover columns that don't fill a full vector (tile edge).
  for (; j < j_end; ++j) {
    crow[j] += aik * brow[j];
  }
}

template <typename T>
__attribute__((target("avx2,fma"))) void
gemm_avx2_impl(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) {
  const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
  const T *a = A.data();
  const T *b = B.data();
  C.fill(T{0});
  T *c = C.data();

  for (std::size_t ii = 0; ii < M; ii += kBS) {
    const std::size_t i_end = std::min(ii + kBS, M);
    for (std::size_t kk = 0; kk < K; kk += kBS) {
      const std::size_t k_end = std::min(kk + kBS, K);
      for (std::size_t jj = 0; jj < N; jj += kBS) {
        const std::size_t j_end = std::min(jj + kBS, N);

        for (std::size_t i = ii; i < i_end; ++i) {
          T *crow = c + i * N;
          for (std::size_t k = kk; k < k_end; ++k) {
            axpy_row<T>(crow, b + k * N, a[i * K + k], jj, j_end);
          }
        }
      }
    }
  }
}

} // namespace

// Public entry point: plain function (no target attribute), so its
// declaration in gemm.hpp matches. It validates, then calls the AVX2 body.
template <typename T>
void gemm_avx2(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) {
  check_gemm_args(A, B, C);
  gemm_avx2_impl<T>(A, B, C);
}

template void gemm_avx2<float>(const Matrix<float> &, const Matrix<float> &,
                               Matrix<float> &);
template void gemm_avx2<double>(const Matrix<double> &, const Matrix<double> &,
                                Matrix<double> &);

} // namespace gemmx
