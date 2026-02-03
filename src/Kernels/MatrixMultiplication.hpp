#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "Math/MatrixView.hpp"

namespace Logos::linalg {
template <class T>
inline void matmul(MatrixView<const T> A, MatrixView<const T> B,
                   MatrixView<T> out) {
  if (A.cols() != B.rows() || out.rows() != A.rows() || out.cols() != B.cols())
    throw std::logic_error("matmul_row_major: shape mismatch");

  if (!A.is_row_major_contiguous() || !B.is_row_major_contiguous() ||
      !out.is_row_major_contiguous())
    throw std::logic_error("matmul_rowmajor: requires row-major contiguous");

  const auto N = A.rows(), K = A.cols(), M = B.cols();
  const auto X = A.data(), Y = B.data();
  auto Z = out.data();
  std::fill(Z, Z + N * M, T{});
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < K; j++) {
      const T val = X[i * K + j];
      for (std::size_t k = 0; k < M; k++)
        Z[i * M + k] += val * Y[j * M + k];
    }
}

template <class T>
inline void matmul_transposeA(MatrixView<const T> A, MatrixView<const T> B,
                              MatrixView<T> out) {
  if (A.rows() != B.rows() || out.rows() != A.cols() || out.cols() != B.cols())
    throw std::logic_error("matmul_transposeA: shape mismatch");
  if (!A.is_row_major_contiguous() || !B.is_row_major_contiguous() ||
      !out.is_row_major_contiguous())
    throw std::logic_error(
        "matmul_transposeA: requires row-major contiguous views");

  // A = [N x M]
  // B = [N x P]
  // out = [M x P]

  const auto N = A.rows(), M = A.cols(), P = B.cols();
  const auto X = A.data(), Y = B.data();
  auto Z = out.data();

  std::fill(Z, Z + M * P, T{});

  for (std::size_t i = 0; i < M; i++)
    for (std::size_t k = 0; k < N; k++) {
      const auto val = X[k * M + i];
      for (std::size_t j = 0; j < P; j++)
        Z[i * P + j] += val * Y[k * P + j];
    }
}
template <class T>
inline void matmul_transposeB(MatrixView<const T> A, MatrixView<const T> B,
                              MatrixView<T> out) {
  if (A.cols() != B.cols() || out.rows() != A.rows() || out.cols() != B.rows())
    throw std::logic_error("matmul_transposeB: shape mismatch");

  if (!A.is_row_major_contiguous() || !B.is_row_major_contiguous() ||
      !out.is_row_major_contiguous())
    throw std::logic_error(
        "matmul_transposeB: requires row-major contiguous views");

  // A = [N x M]
  // B = [P x M]
  // out = [N x P]
  const auto N = A.rows(), M = A.cols(), P = B.rows();
  const auto X = A.data(), Y = B.data();
  auto Z = out.data();
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < P; j++) {
      T sum{0};
      for (std::size_t k = 0; k < M; k++)
        sum += X[i * M + k] * Y[j * M + k];
      Z[i * P + j] = sum;
    }
}
} // namespace Logos::linalg
