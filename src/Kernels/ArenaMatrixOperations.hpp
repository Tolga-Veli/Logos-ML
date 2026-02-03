#pragma once

#include <stdexcept>

#include "Math/ArenaMatrix.hpp"
#include "MatrixAddition.hpp"
#include "MatrixMultiplication.hpp"

namespace Logos::linalg {
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

template <class T>
inline void matmul_transposeA(const ArenaMatrix<T> &A, const ArenaMatrix<T> &B,
                              ArenaMatrix<T> &out) {
  if (A.rows() != B.rows())
    throw std::logic_error("matmul_transposeA: shape mismatch");

  const auto M = A.cols(), P = B.cols();
  if (out.rows() != M || out.cols() != P)
    out.allocate(M, P);
  else
    out.fill_zeroes();

  matmul_transposeA(A.view(), B.view(), out.view());
}

template <class T>
inline void matmul_transposeB(const ArenaMatrix<T> &A, const ArenaMatrix<T> &B,
                              ArenaMatrix<T> &out) {
  if (A.cols() != B.cols())
    throw std::logic_error("matmul_transposeB: shape mismatch");

  const auto N = A.rows(), P = B.rows();
  if (out.rows() != N || out.cols() != P)
    out.allocate(N, P);

  matmul_transposeB(A.view(), B.view(), out.view());
}

template <class T>
inline void add_rowwise_bias(const std::vector<T> &b, ArenaMatrix<T> &out) {
  add_rowwise_bias(b, out.view());
}

template <class T>
inline void sum_rows(const ArenaMatrix<T> &A, std::vector<T> &out) {
  sum_rows(A.cview(), out);
}
} // namespace Logos::linalg
