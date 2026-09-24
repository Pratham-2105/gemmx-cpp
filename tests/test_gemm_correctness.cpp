#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "gemmx/gemm.hpp"

using gemmx::Matrix;

namespace {

// ---------------------------------------------------------------------------
// Kernel registry. Every new kernel gets ONE line here, and then every test
// in this file runs against it automatically.
// ---------------------------------------------------------------------------
template <typename T>
using KernelFn = void (*)(const Matrix<T> &, const Matrix<T> &, Matrix<T> &);

template <typename T> struct NamedKernel {
  const char *name;
  KernelFn<T> fn;
};

template <typename T> std::vector<NamedKernel<T>> all_kernels() {
  return {
      {"reference", &gemmx::gemm_reference<T>},
  };
}

// ---------------------------------------------------------------------------
// Oracle: the same product accumulated in double, plus (|A| |B|)_ij, which is
// what the rounding-error bound scales with.
// ---------------------------------------------------------------------------
template <typename T>
void oracle(const Matrix<T> &A, const Matrix<T> &B, std::vector<double> &exact,
            std::vector<double> &abs_prod) {
  const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
  exact.assign(M * N, 0.0);
  abs_prod.assign(M * N, 0.0);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      double s = 0.0, sa = 0.0;
      for (std::size_t k = 0; k < K; ++k) {
        const double p =
            static_cast<double>(A(i, k)) * static_cast<double>(B(k, j));
        s += p;
        sa += std::abs(p);
      }
      exact[i * N + j] = s;
      abs_prod[i * N + j] = sa;
    }
  }
}

// Per-element check against the standard dot-product error bound
//   |computed - exact| <= gamma_K * (|A||B|)_ij,   gamma_K ~= K * eps / 2
// (Higham, "Accuracy and Stability of Numerical Algorithms", Sec. 3.1).
// The bound holds for ANY summation order, so it stays valid for blocked,
// SIMD, and multithreaded kernels that add the terms in a different order.
// We allow 2 * K * eps: a 2x margin for double (where the oracle itself
// rounds too) and more for float.
template <typename T>
void expect_matches_oracle(const Matrix<T> &A, const Matrix<T> &B,
                           const Matrix<T> &C, const char *kernel_name) {
  std::vector<double> exact, abs_prod;
  oracle(A, B, exact, abs_prod);

  const double eps = std::numeric_limits<T>::epsilon();
  const double K = static_cast<double>(A.cols());
  const std::size_t N = C.cols();

  std::size_t failures = 0;
  for (std::size_t i = 0; i < C.rows(); ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      const std::size_t idx = i * N + j;
      const double err = std::abs(static_cast<double>(C(i, j)) - exact[idx]);
      const double tol = 2.0 * K * eps * abs_prod[idx];
      if (!(err <= tol)) { // written as !(<=) so a NaN also counts as a failure
        if (++failures <= 5) {
          ADD_FAILURE() << kernel_name << ": C(" << i << "," << j
                        << ") = " << C(i, j) << ", expected " << exact[idx]
                        << ", |err| = " << err << " > tol = " << tol;
        }
      }
    }
  }
  EXPECT_EQ(failures, 0u) << kernel_name << ": " << failures
                          << " element(s) out of tolerance";
}

} // namespace

// ---------------------------------------------------------------------------
// Every test below runs once for float and once for double.
// ---------------------------------------------------------------------------
template <typename T> class GemmCorrectness : public ::testing::Test {};
using ElementTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GemmCorrectness, ElementTypes);

// Small enough to verify with pencil and paper. Non-square on purpose:
// square inputs hide bugs where i/j or rows/cols get swapped.
TYPED_TEST(GemmCorrectness, HandComputed2x3Times3x2) {
  using T = TypeParam;
  Matrix<T> A(2, 3), B(3, 2);
  const T a[] = {1, 2, 3, 4, 5, 6};
  const T b[] = {7, 8, 9, 10, 11, 12};
  std::copy(std::begin(a), std::end(a), A.data());
  std::copy(std::begin(b), std::end(b), B.data());
  const T expected[] = {58, 64, 139, 154};

  for (const auto &k : all_kernels<T>()) {
    Matrix<T> C(2, 2, T{-999});
    k.fn(A, B, C);
    for (std::size_t idx = 0; idx < 4; ++idx) {
      EXPECT_EQ(C.data()[idx], expected[idx]) << k.name << " at " << idx;
    }
  }
}

// A * I must reproduce A exactly: each output is a single a*1 plus exact zeros.
TYPED_TEST(GemmCorrectness, RightIdentityIsExact) {
  using T = TypeParam;
  Matrix<T> A(5, 7), I(7, 7);
  gemmx::fill_random(A, 11);
  for (std::size_t d = 0; d < 7; ++d)
    I(d, d) = T{1};

  for (const auto &k : all_kernels<T>()) {
    Matrix<T> C(5, 7);
    k.fn(A, I, C);
    for (std::size_t idx = 0; idx < A.size(); ++idx) {
      EXPECT_EQ(C.data()[idx], A.data()[idx]) << k.name << " at " << idx;
    }
  }
}

// Random inputs over awkward shapes: 1-sized dimensions, primes, and sizes
// that are one past a power of two (these will later be non-multiples of the
// block size and SIMD width, which is where tiled kernels break).
TYPED_TEST(GemmCorrectness, RandomShapes) {
  using T = TypeParam;
  struct Shape {
    std::size_t M, K, N;
  };
  const Shape shapes[] = {
      {1, 1, 1},       {1, 17, 1},      {3, 1, 4},    {7, 13, 5},
      {16, 16, 16},    {33, 65, 17},    {64, 64, 64}, {100, 37, 129},
      {128, 128, 128}, {256, 256, 256},
  };

  std::uint64_t seed = 1000;
  for (const Shape &s : shapes) {
    SCOPED_TRACE("M=" + std::to_string(s.M) + " K=" + std::to_string(s.K) +
                 " N=" + std::to_string(s.N));
    Matrix<T> A(s.M, s.K), B(s.K, s.N);
    gemmx::fill_random(A, seed++);
    gemmx::fill_random(B, seed++);

    for (const auto &k : all_kernels<T>()) {
      // Pre-fill with NaN: if a kernel skips any element, the test catches it.
      Matrix<T> C(s.M, s.N, std::numeric_limits<T>::quiet_NaN());
      k.fn(A, B, C);
      expect_matches_oracle(A, B, C, k.name);
    }
  }
}

TYPED_TEST(GemmCorrectness, ShapeMismatchThrows) {
  using T = TypeParam;
  Matrix<T> A(2, 3), B(4, 2), C(2, 2); // A.cols() != B.rows()
  for (const auto &k : all_kernels<T>()) {
    EXPECT_THROW(k.fn(A, B, C), std::invalid_argument) << k.name;
  }
}

TYPED_TEST(GemmCorrectness, AliasingThrows) {
  using T = TypeParam;
  Matrix<T> A(4, 4), B(4, 4);
  for (const auto &k : all_kernels<T>()) {
    EXPECT_THROW(k.fn(A, B, A), std::invalid_argument) << k.name;
    EXPECT_THROW(k.fn(A, B, B), std::invalid_argument) << k.name;
  }
}
