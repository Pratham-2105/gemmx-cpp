#include <algorithm>
#include <cstddef>
#include <immintrin.h>
#include <new>

#include "gemmx/gemm.hpp"

namespace gemmx {
namespace {

// ---- Block sizes (from docs/cpu_caches.txt, P-core values) -----------------
constexpr std::size_t kMR = 6; // rows of C per microkernel tile
constexpr std::size_t kKC =
    256; // k-depth of one pass  (A+B micro-panels in L1)
constexpr std::size_t kMC = 288;  // rows of packed A     (packed A block in L2)
constexpr std::size_t kNC = 4096; // cols of packed B     (packed B block in L3)
static_assert(kMC % kMR == 0, "MC must be a multiple of MR");

// ---- 64-byte aligned scratch buffer (freed automatically) ------------------
template <typename T> class AlignedBuf {
public:
  explicit AlignedBuf(std::size_t n)
      : p_(static_cast<T *>(
            ::operator new[](n * sizeof(T), std::align_val_t{64}))) {}
  ~AlignedBuf() { ::operator delete[](p_, std::align_val_t{64}); }
  AlignedBuf(const AlignedBuf &) = delete;
  AlignedBuf &operator=(const AlignedBuf &) = delete;
  T *get() const { return p_; }

private:
  T *p_;
};

std::size_t round_up(std::size_t x, std::size_t m) {
  return (x + m - 1) / m * m;
}

// ===========================================================================
// Everything between push_options and pop_options is compiled with AVX2+FMA.
// It only ever runs after the registry's cpu_has_avx2_fma() check.
// ===========================================================================
#pragma GCC push_options
#pragma GCC target("avx2,fma")

// One type-generic wrapper so the microkernel is written once for both types.
template <typename T> struct Simd;

template <> struct Simd<float> {
  using V = __m256;
  static constexpr std::size_t W = 8; // lanes per register
  static V zero() { return _mm256_setzero_ps(); }
  static V load(const float *p) { return _mm256_loadu_ps(p); }
  static V bcast(const float *p) { return _mm256_broadcast_ss(p); }
  static V fma(V a, V b, V c) { return _mm256_fmadd_ps(a, b, c); }
  static V add(V a, V b) { return _mm256_add_ps(a, b); }
  static void store(float *p, V v) { _mm256_storeu_ps(p, v); }
};

template <> struct Simd<double> {
  using V = __m256d;
  static constexpr std::size_t W = 4;
  static V zero() { return _mm256_setzero_pd(); }
  static V load(const double *p) { return _mm256_loadu_pd(p); }
  static V bcast(const double *p) { return _mm256_broadcast_sd(p); }
  static V fma(V a, V b, V c) { return _mm256_fmadd_pd(a, b, c); }
  static V add(V a, V b) { return _mm256_add_pd(a, b); }
  static void store(double *p, V v) { _mm256_storeu_pd(p, v); }
};

// NR = two registers wide: 16 floats or 8 doubles.
template <typename T> constexpr std::size_t kNR = 2 * Simd<T>::W;

// ---- Packing ---------------------------------------------------------------
// A block (mc x kc, starting at a, row stride lda) -> panels of MR rows.
// Layout per panel: for each k, the MR values of that column sit together.
// Rows past mc are padded with zeros.
template <typename T>
void pack_a(const T *a, std::size_t lda, std::size_t mc, std::size_t kc,
            T *buf) {
  for (std::size_t ir = 0; ir < mc; ir += kMR) {
    const std::size_t rows = std::min(kMR, mc - ir);
    for (std::size_t k = 0; k < kc; ++k) {
      for (std::size_t r = 0; r < kMR; ++r) {
        *buf++ = (r < rows) ? a[(ir + r) * lda + k] : T{0};
      }
    }
  }
}

// B block (kc x nc, starting at b, row stride ldb) -> panels of NR columns.
// Layout per panel: for each k, the NR values of that row sit together.
// Columns past nc are padded with zeros.
template <typename T>
void pack_b(const T *b, std::size_t ldb, std::size_t kc, std::size_t nc,
            T *buf) {
  constexpr std::size_t NR = kNR<T>;
  for (std::size_t jr = 0; jr < nc; jr += NR) {
    const std::size_t cols = std::min(NR, nc - jr);
    for (std::size_t k = 0; k < kc; ++k) {
      const T *src = b + k * ldb + jr;
      for (std::size_t c = 0; c < NR; ++c) {
        *buf++ = (c < cols) ? src[c] : T{0};
      }
    }
  }
}

// ---- Microkernel -----------------------------------------------------------
// C[MR x NR] += Apanel[MR x kc] * Bpanel[kc x NR]
// The whole C tile lives in 12 registers for all kc steps.
template <typename T>
void microkernel(std::size_t kc, const T *a, const T *b, T *c,
                 std::size_t ldc) {
  using S = Simd<T>;
  constexpr std::size_t W = S::W;
  typename S::V acc[kMR][2];

#pragma GCC unroll 6
  for (std::size_t r = 0; r < kMR; ++r) {
    acc[r][0] = S::zero();
    acc[r][1] = S::zero();
  }

  for (std::size_t k = 0; k < kc; ++k) {
    const auto b0 = S::load(b);     // B[k, 0:W]
    const auto b1 = S::load(b + W); // B[k, W:2W]
#pragma GCC unroll 6
    for (std::size_t r = 0; r < kMR; ++r) {
      const auto ar = S::bcast(a + r); // A[r, k] in every lane
      acc[r][0] = S::fma(ar, b0, acc[r][0]);
      acc[r][1] = S::fma(ar, b1, acc[r][1]);
    }
    a += kMR;   // next column of the A panel
    b += 2 * W; // next row of the B panel
  }

  // Touch C once: C += accumulated tile.
#pragma GCC unroll 6
  for (std::size_t r = 0; r < kMR; ++r) {
    T *crow = c + r * ldc;
    S::store(crow, S::add(S::load(crow), acc[r][0]));
    S::store(crow + W, S::add(S::load(crow + W), acc[r][1]));
  }
}

// ---- The 5 loops -----------------------------------------------------------
template <typename T>
void gemm_packed_impl(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) {
  constexpr std::size_t NR = kNR<T>;
  static_assert(kNC % NR == 0, "NC must be a multiple of NR");

  const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
  C.fill(T{0});
  if (M == 0 || N == 0 || K == 0)
    return;

  const T *a = A.data();
  const T *b = B.data();
  T *c = C.data();

  // Buffers sized to what this problem actually needs (with padding).
  AlignedBuf<T> abuf(round_up(std::min(M, kMC), kMR) * std::min(K, kKC));
  AlignedBuf<T> bbuf(round_up(std::min(N, kNC), NR) * std::min(K, kKC));

  for (std::size_t jc = 0; jc < N; jc += kNC) { // loop 5: NC
    const std::size_t nc = std::min(kNC, N - jc);
    for (std::size_t pc = 0; pc < K; pc += kKC) { // loop 4: KC
      const std::size_t kc = std::min(kKC, K - pc);
      pack_b(b + pc * N + jc, N, kc, nc, bbuf.get());

      for (std::size_t ic = 0; ic < M; ic += kMC) { // loop 3: MC
        const std::size_t mc = std::min(kMC, M - ic);
        pack_a(a + ic * K + pc, K, mc, kc, abuf.get());

        for (std::size_t jr = 0; jr < nc; jr += NR) { // loop 2: NR
          const std::size_t cols = std::min(NR, nc - jr);
          const T *bp = bbuf.get() + jr * kc; // B micro-panel

          for (std::size_t ir = 0; ir < mc; ir += kMR) { // loop 1: MR
            const std::size_t rows = std::min(kMR, mc - ir);
            const T *ap = abuf.get() + ir * kc; // A micro-panel
            T *cp = c + (ic + ir) * N + (jc + jr);

            if (rows == kMR && cols == NR) {
              microkernel(kc, ap, bp, cp, N); // full tile: straight into C
            } else {
              alignas(64) T tmp[kMR * NR] = {}; // edge tile: go via temp
              microkernel(kc, ap, bp, tmp, NR);
              for (std::size_t r = 0; r < rows; ++r)
                for (std::size_t j = 0; j < cols; ++j)
                  cp[r * N + j] += tmp[r * NR + j];
            }
          }
        }
      }
    }
  }
}

#pragma GCC pop_options

} // namespace

// Public entry point: plain (non-AVX) function matching gemm.hpp.
template <typename T>
void gemm_packed(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C) {
  check_gemm_args(A, B, C);
  gemm_packed_impl<T>(A, B, C);
}

template void gemm_packed<float>(const Matrix<float> &, const Matrix<float> &,
                                 Matrix<float> &);
template void gemm_packed<double>(const Matrix<double> &,
                                  const Matrix<double> &, Matrix<double> &);

} // namespace gemmx
