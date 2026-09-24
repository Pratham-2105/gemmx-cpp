#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <random>
#include <vector>

namespace gemmx {

// 64 bytes = one x86 cache line, and a multiple of the 32-byte AVX2 register.
inline constexpr std::size_t kAlignment = 64;

// Minimal allocator so std::vector hands out 64-byte-aligned memory.
// Uses C++17 aligned operator new, which works on both GCC and MSVC
// (std::aligned_alloc does not exist on MSVC).
template <typename T, std::size_t Align = kAlignment> struct AlignedAllocator {
  using value_type = T;

  // Needed because Align is a non-type template parameter, which
  // std::allocator_traits cannot rebind automatically.
  template <typename U> struct rebind {
    using other = AlignedAllocator<U, Align>;
  };

  AlignedAllocator() noexcept = default;
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Align> &) noexcept {}

  T *allocate(std::size_t n) {
    if (n > static_cast<std::size_t>(-1) / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    return static_cast<T *>(
        ::operator new(n * sizeof(T), std::align_val_t{Align}));
  }

  void deallocate(T *p, std::size_t) noexcept {
    ::operator delete(p, std::align_val_t{Align});
  }
};

template <typename T, typename U, std::size_t A>
bool operator==(const AlignedAllocator<T, A> &,
                const AlignedAllocator<U, A> &) noexcept {
  return true; // stateless: any instance can free memory from any other
}

// Dense row-major matrix: element (i, j) lives at data()[i * cols() + j].
template <typename T> class Matrix {
public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols, T init = T{0})
      : rows_(rows), cols_(cols), data_(rows * cols, init) {}

  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }
  std::size_t size() const noexcept { return data_.size(); }

  // Unchecked on purpose: these sit inside the hot loops.
  T &operator()(std::size_t i, std::size_t j) noexcept {
    return data_[i * cols_ + j];
  }
  const T &operator()(std::size_t i, std::size_t j) const noexcept {
    return data_[i * cols_ + j];
  }

  T *data() noexcept { return data_.data(); }
  const T *data() const noexcept { return data_.data(); }

  void fill(T value) { std::fill(data_.begin(), data_.end(), value); }

private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<T, AlignedAllocator<T>> data_;
};

// Fills m with values in [-1, 1], identically on every compiler and OS.
// std::mt19937_64's output sequence is fixed by the C++ standard, but
// std::uniform_real_distribution's is NOT (libstdc++ and MSVC's STL differ),
// so we turn the raw 64-bit outputs into doubles ourselves.
// (For float, the top value 1 - 2^-52 rounds to 1.0f, hence the closed range.)
template <typename T> void fill_random(Matrix<T> &m, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  T *p = m.data();
  for (std::size_t i = 0; i < m.size(); ++i) {
    const double u01 = static_cast<double>(rng() >> 11) * 0x1.0p-53; // [0, 1)
    p[i] = static_cast<T>(2.0 * u01 - 1.0);
  }
}

} // namespace gemmx
