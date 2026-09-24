#include <gtest/gtest.h>

#include <cstdint>

#include "gemmx/matrix.hpp"

using gemmx::Matrix;

TEST(Matrix, StorageIsAligned) {
  for (std::size_t n : {1u, 3u, 17u, 64u, 1000u}) {
    Matrix<float> mf(n, n);
    Matrix<double> md(n, n);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(mf.data()) % gemmx::kAlignment,
              0u)
        << n;
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(md.data()) % gemmx::kAlignment,
              0u)
        << n;
  }
}

TEST(Matrix, RowMajorIndexing) {
  Matrix<float> m(3, 4);
  m(1, 2) = 42.0f;
  EXPECT_EQ(m.data()[1 * 4 + 2], 42.0f);
  EXPECT_EQ(m.rows(), 3u);
  EXPECT_EQ(m.cols(), 4u);
  EXPECT_EQ(m.size(), 12u);
}

TEST(Matrix, FillRandomIsDeterministic) {
  Matrix<double> a(8, 8), b(8, 8), c(8, 8);
  gemmx::fill_random(a, 123);
  gemmx::fill_random(b, 123);
  gemmx::fill_random(c, 124);

  bool any_diff = false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a.data()[i], b.data()[i]); // same seed -> same data
    if (a.data()[i] != c.data()[i])
      any_diff = true;
  }
  EXPECT_TRUE(any_diff); // different seed -> different data
}

TEST(Matrix, FillRandomRange) {
  Matrix<float> m(64, 64);
  gemmx::fill_random(m, 7);
  for (std::size_t i = 0; i < m.size(); ++i) {
    EXPECT_GE(m.data()[i], -1.0f);
    EXPECT_LE(m.data()[i], 1.0f);
  }
}
