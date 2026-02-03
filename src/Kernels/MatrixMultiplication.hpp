#pragma once

#include <cstddef>
#include <stdexcept>

#include "Math/ArenaMatrix.hpp"
#include "Math/Matrix.hpp"
#include "Math/MatrixView.hpp"

namespace Logos::linalg {

template <class T>
inline void matmul_row_major(MatrixView<const T> A, MatrixView<const T> B,
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
inline void matmul_col_major(MatrixView<const T> A, MatrixView<const T> B,
                             MatrixView<T> out) {
  if (A.cols() != B.rows() || out.rows() != A.rows() || out.cols() != B.cols())
    throw std::logic_error("matmul_col_major: shape mismatch");

  if (!A.is_col_major_contiguous() || !B.is_col_major_contiguous() ||
      !out.is_col_major_contiguous())
    throw std::logic_error("matmul_col_major: requires col-major contiguous");

  const auto N = A.rows(), K = A.cols(), M = B.cols();
  const auto X = A.data(), Y = B.data();
  auto Z = out.data();
  std::fill(Z, Z + N * M, T{});
  for (std::size_t k = 0; k < M; k++)
    for (std::size_t j = 0; j < K; j++) {
      const T val = Y[k * K + j];
      for (std::size_t i = 0; i < N; i++)
        Z[k * N + i] += val * X[j * N + i];
    }
}

template <class T>
inline void matmul(MatrixView<const T> A, MatrixView<const T> B,
                   MatrixView<T> out) {
  if (A.cols() != B.rows() || out.rows() != A.rows() ||
      out.cols() != B.cols()) {
    throw std::logic_error("matmul: shape mismatch");
  }

  if (A.is_row_major_contiguous() && B.is_row_major_contiguous() &&
      out.is_row_major_contiguous())
    matmul_row_major(A, B, out);

  else if (A.is_col_major_contiguous() && B.is_col_major_contiguous() &&
           out.is_col_major_contiguous())
    matmul_col_major(A, B, out);
  else {
    // generic implementation fallback
    if (A.cols() != B.rows() || out.rows() != A.rows() ||
        out.cols() != B.cols())
      throw std::logic_error("matmul_generic: shape mismatch");

    const std::size_t N = A.rows(), K = A.cols(), M = B.cols();
    for (std::size_t i = 0; i < N; i++)
      for (std::size_t j = 0; j < M; j++)
        out(i, j) = T{};

    for (std::size_t i = 0; i < N; i++)
      for (std::size_t k = 0; k < K; k++) {
        const T a = A(i, k);
        for (std::size_t j = 0; j < M; j++)
          out(i, j) += a * B(k, j);
      }
  }
}

template <class T>
inline void matmul(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &out) {
  if (A.cols() != B.rows())
    throw std::logic_error("matmul(Matrix): shape mismatch");

  const auto N = A.rows(), M = B.cols();
  if (out.rows() != N || out.cols() != M)
    out = Matrix<T>(N, M);
  else
    out.fill_zeroes();

  matmul(A.cview(), B.cview(), out.view());
}

template <class T>
inline void matmul(const ArenaMatrix<T> &A, const ArenaMatrix<T> &B,
                   ArenaMatrix<T> &out) {
  if (A.cols() != B.rows())
    throw std::logic_error("matmul(ArenaMatrix): shape mismatch");

  const auto N = A.rows(), M = B.cols();
  if (out.rows() != N || out.cols() != M)
    out.allocate(N, M);
  else
    out.fill_zeroes();

  matmul(A.cview(), B.cview(), out.view());
}
} // namespace Logos::linalg
