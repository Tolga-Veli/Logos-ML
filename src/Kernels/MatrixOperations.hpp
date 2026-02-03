#pragma once

#include "Math/Matrix.hpp"
#include <vector>

namespace Logos::linalg {
template <class T>
inline void add_rowwise_bias(const std::vector<T> &b, Matrix<T> &out) {
  add_rowwise_bias(b, out.view());
}

template <class T>
inline void sum_rows(const Matrix<T> &A, std::vector<T> &out) {
  sum_rows(A.cview(), out);
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
} // namespace Logos::linalg
