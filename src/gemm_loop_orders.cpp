#include "gemmx/gemm.hpp"

namespace gemmx {

// All six orders compute the same C = A*B; only the nesting of i, j, k changes.
// Every kernel uses the same raw-pointer addressing so that nesting is the
// ONLY difference between them (row-major):
//   A(i,k) = a[i*K + k]    B(k,j) = b[k*N + j]    C(i,j) = c[i*N + j]
//
// Orders whose inner loop is NOT k accumulate into C, so they zero C first.

// j-i-k: like the reference, but walks C column by column.
// Inner loop: A along a row, B down a column, single scalar reduction.
template <typename T>
void gemm_loop_jik(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) {
  check_gemm_args(A, B, C);
  const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
  const T *a = A.data();
  const T *b = B.data();
  T *c = C.data();

  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < M; ++i) {
      T sum = T{0};
      for (std::size_t k = 0; k < K; ++k) {
        sum += a[i * K + k] * b[k * N + j];
      }
      c[i * N + j] = sum;
    }
  }
}

// i-k-j: inner loop streams a row of B into a row of C. No reduction,
// so the compiler can vectorize it.
template <typename T>
void gemm_loop_ikj(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) {
  check_gemm_args(A, B, C);
  const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
  const T *a = A.data();
  const T *b = B.data();
  C.fill(T{0});
  T *c = C.data();

  for (std::size_t i = 0; i < M; ++i) {
    T *crow = c + i * N;
    for (std::size_t k = 0; k < K; ++k) {
      const T aik = a[i * K + k];
      const T *brow = b + k * N;
      for (std::size_t j = 0; j < N; ++j) {
        crow[j] += aik * brow[j];
      }
    }
  }
}

// k-i-j: same inner loop as i-k-j, but k outermost: each row of B is
// applied to every row of C before moving to the next row of B.
template <typename T>
void gemm_loop_kij(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) {
  check_gemm_args(A, B, C);
  const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
  const T *a = A.data();
  const T *b = B.data();
  C.fill(T{0});
  T *c = C.data();

  for (std::size_t k = 0; k < K; ++k) {
    const T *brow = b + k * N;
    for (std::size_t i = 0; i < M; ++i) {
      const T aik = a[i * K + k];
      T *crow = c + i * N;
      for (std::size_t j = 0; j < N; ++j) {
        crow[j] += aik * brow[j];
      }
    }
  }
}

// j-k-i: inner loop walks DOWN a column of A and a column of C.
template <typename T>
void gemm_loop_jki(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) {
  check_gemm_args(A, B, C);
  const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
  const T *a = A.data();
  const T *b = B.data();
  C.fill(T{0});
  T *c = C.data();

  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t k = 0; k < K; ++k) {
      const T bkj = b[k * N + j];
      for (std::size_t i = 0; i < M; ++i) {
        c[i * N + j] += a[i * K + k] * bkj;
      }
    }
  }
}

// k-j-i: same inner loop as j-k-i, with k outermost.
template <typename T>
void gemm_loop_kji(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) {
  check_gemm_args(A, B, C);
  const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
  const T *a = A.data();
  const T *b = B.data();
  C.fill(T{0});
  T *c = C.data();

  for (std::size_t k = 0; k < K; ++k) {
    for (std::size_t j = 0; j < N; ++j) {
      const T bkj = b[k * N + j];
      for (std::size_t i = 0; i < M; ++i) {
        c[i * N + j] += a[i * K + k] * bkj;
      }
    }
  }
}

#define GEMMX_INSTANTIATE(fn)                                                  \
  template void fn<float>(const Matrix<float> &, const Matrix<float> &,        \
                          Matrix<float> &);                                    \
  template void fn<double>(const Matrix<double> &, const Matrix<double> &,     \
                           Matrix<double> &);

GEMMX_INSTANTIATE(gemm_loop_ikj)
GEMMX_INSTANTIATE(gemm_loop_jik)
GEMMX_INSTANTIATE(gemm_loop_jki)
GEMMX_INSTANTIATE(gemm_loop_kij)
GEMMX_INSTANTIATE(gemm_loop_kji)

#undef GEMMX_INSTANTIATE

} // namespace gemmx
