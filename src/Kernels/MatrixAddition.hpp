#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "Math/MatrixView.hpp"

namespace Logos::linalg {

template <class T>
inline void add_rowwise_bias(const std::vector<T> &b, MatrixView<T> out) {
  if (b.size() != out.cols())
    throw std::logic_error("add_rowwise_bias: size mismatch");

  if (!out.is_row_major_contiguous())
    throw std::logic_error(
        "add_rowwise_bias: row-major matrix layout expected!");

  const auto N = out.rows(), M = out.cols();
  auto X = out.data();
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < M; j++)
      X[i * M + j] += b[j];
}

template <class T>
inline void sum_rows(MatrixView<const T> A, std::vector<T> &out) {
  if (!A.is_row_major_contiguous())
    throw std::logic_error("sum_rows_row: row-major matrix layout expected!");

  out.assign(A.cols(), T{});
  const auto N = A.rows(), M = A.cols();
  const auto X = A.data();
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < M; j++)
      out[j] += X[i * M + j];
}

} // namespace Logos::linalg
