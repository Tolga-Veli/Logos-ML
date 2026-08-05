#pragma once

#include "BlasOps.hpp"

#include <cblas.h>

namespace ml::linalg {
// C := alpha * op(A) * op(B) + beta * C   (written into `out`, i.e. C)
//
// op(A) is transA==Yes ? A^T : A,  shape M x K
// op(B) is transB==Yes ? B^T : B,  shape K x N
// out (C) must be shape M x N
//
// M, K come from A (adjusted for transA); N, K come from B (adjusted for
// transB) -- K must agree from both sides, checked by the assert below.
template <class T>
void matmul_impl(Transpose transA, Transpose transB, T alpha,
                 MatrixView<const T> A, MatrixView<const T> B, T beta,
                 MatrixView<T> out) {
  const auto opA = (transA == Transpose::Yes ? CblasTrans : CblasNoTrans);
  const auto opB = (transB == Transpose::Yes ? CblasTrans : CblasNoTrans);

  const auto M = (transA == Transpose::Yes ? A.cols() : A.rows());
  const auto K = (transA == Transpose::Yes ? A.rows() : A.cols());
  const auto N = (transB == Transpose::Yes ? B.rows() : B.cols());

  // Shared contraction dimension must match on both operands.
  assert(K == (transB == Transpose::Yes ? B.cols() : B.rows()));
  // Output shape must match M x N derived above.
  assert(out.rows() == M && out.cols() == N);

  detail::BlasDispatch<T>::gemm(CblasRowMajor, opA, opB, M, N, K, alpha,
                                A.data(), A.ld(), B.data(), B.ld(), beta,
                                out.data(), out.ld());
}

template <class T>
void matmul(Transpose transA, Transpose transB, T alpha, const Tensor<T> &A,
            const Tensor<T> &B, T beta, Tensor<T> &out) {
  assert(A.rank() == 2 && B.rank() == 2 && out.rank() == 2);
  matmul_impl(transA, transB, alpha, core::as_matrix(A), core::as_matrix(B),
              beta, core::as_matrix(out));
}

// C = op(A) * op(B) (alpha = 1, beta = 0, i.e. out is fully overwritten).
template <class T>
void matmul(Transpose transA, Transpose transB, const Tensor<T> &A,
            const Tensor<T> &B, Tensor<T> &out) {
  matmul(transA, transB, T{1}, A, B, T{0}, out);
}
} // namespace ml::linalg
