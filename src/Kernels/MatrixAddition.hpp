#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Math/ArenaMatrix.hpp"
#include "Math/Matrix.hpp"
#include "Math/MatrixView.hpp"

namespace Logos::linalg {

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
inline void add_rowwise_bias(const std::vector<T> &b, Matrix<T> &out) {
  add_rowwise_bias(b, out.view());
}

template <class T>
inline void add_rowwise_bias(const std::vector<T> &b, ArenaMatrix<T> &out) {
  add_rowwise_bias(b, out.view());
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

template <class T>
inline void sum_rows(const Matrix<T> &A, std::vector<T> &out) {
  sum_rows(A.cview(), out);
}

template <class T>
inline void sum_rows(const ArenaMatrix<T> &A, std::vector<T> &out) {
  sum_rows(A.cview(), out);
}
} // namespace Logos::linalg
