#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

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
inline void add_rowwise_bias_row_major(const std::vector<T> &b,
                                       MatrixView<T> out) {
  if (b.size() != out.cols())
    throw std::logic_error("add_rowwise_bias_row_major: size mismatch");

  if (!out.is_row_major_contiguous())
    throw std::logic_error(
        "add_rowwise_bias_row_major: row-major matrix layout expected!");

  const auto N = out.rows(), M = out.cols();
  auto X = out.data();
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < M; j++)
      X[i * M + j] += b[j];
}

template <class T>
inline void add_rowwise_bias_col_major(const std::vector<T> &b,
                                       MatrixView<T> out) {
  if (b.size() != out.cols())
    throw std::logic_error("add_rowwise_bias_col_major: size mismatch");

  if (!out.is_col_major_contiguous())
    throw std::logic_error(
        "add_rowwise_bias_col_major: col-major matrix layout expected!");

  const auto N = out.rows(), M = out.cols();
  auto X = out.data();
  for (std::size_t j = 0; j < M; j++)
    for (std::size_t i = 0; i < N; i++)
      X[j * N + i] += b[j];
}

template <class T>
inline void add_rowwise_bias(const std::vector<T> &b, MatrixView<T> out) {
  if (out.is_row_major_contiguous())
    add_rowwise_bias_row_major(b, out);
  else if (out.is_col_major_contiguous())
    add_rowwise_bias_col_major(b, out);
  else
    throw std::logic_error("add_rowwise_bias: unsupported matrix layout");
}

template <class T>
inline void sum_rows_row_major(MatrixView<const T> A, std::vector<T> &out) {
  if (!A.is_row_major_contiguous())
    throw std::logic_error(
        "sum_rows_row_major: row-major matrix layout expected!");

  out.assign(A.cols(), T{});
  const auto N = A.rows(), M = A.cols();
  const auto X = A.data();
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < M; j++)
      out[j] += X[i * M + j];
}

template <class T>
inline void sum_rows_col_major(MatrixView<const T> A, std::vector<T> &out) {
  if (!A.is_col_major_contiguous())
    throw std::logic_error(
        "sum_rows_col_major: col-major matrix layout expected!");

  out.assign(A.cols(), T{});
  const auto N = A.rows(), M = A.cols();
  const auto X = A.data();
  for (std::size_t j = 0; j < M; j++)
    for (std::size_t i = 0; i < N; i++)
      out[j] += X[j * N + i];
}

template <class T>
inline void sum_rows(MatrixView<const T> A, std::vector<T> &out) {
  if (A.is_row_major_contiguous())
    sum_rows_row_major(A, out);
  else if (A.is_col_major_contiguous())
    sum_rows_col_major(A, out);
  else
    throw std::logic_error("sum_rows: unsupported matrix layout");
}

} // namespace Logos::linalg
