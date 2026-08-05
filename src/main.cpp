#pragma once

#include "Core/MatrixView.hpp"
#include "Math/BlasOps.hpp"

#include <cblas.h>

namespace ml::linalg {

template <class T> using MatrixView = core::MatrixView<T>;

// C := alpha * op(A) * op(B) + beta * C   (written into `out`, i.e. C)
//
// op(A) is transA==Yes ? A^T : A,  shape M x K
// op(B) is transB==Yes ? B^T : B,  shape K x N
// out (C) must be shape M x N
//
// M, K come from A (adjusted for transA); N, K come from B (adjusted for
// transB) -- K must agree from both sides, checked by the assert below.
template <class T>
void matmul(Transpose transA, Transpose transB, T alpha, MatrixView<const T> A,
            MatrixView<const T> B, T beta, MatrixView<T> out) {
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

// Convenience: C = op(A) * op(B)   (alpha=1, beta=0, i.e. out is fully
// overwritten). This is the common case; use the full overload when you need
// accumulation.
template <class T>
void matmul(Transpose transA, Transpose transB, MatrixView<const T> A,
            MatrixView<const T> B, MatrixView<T> out) {
  matmul(transA, transB, T{1}, A, B, T{0}, out);
}

// C := alpha * A * A^T + beta * C   (trans == No)
// C := alpha * A^T * A + beta * C   (trans == Yes)
// C must be square (N x N); only the `uplo` triangle of C is written --
// the other triangle is left untouched by BLAS, so a freshly-allocated
// `out` will have garbage in the unwritten half unless you fill it or
// mirror it yourself afterward.
template <class T>
void syrk(Triangular tri, Transpose trans, T alpha, MatrixView<const T> A,
          T beta, MatrixView<T> out) {
  const auto ul = (tri == Triangular::Upper ? CblasUpper : CblasLower);
  const auto op = (trans == Transpose::Yes ? CblasTrans : CblasNoTrans);

  const auto N = (trans == Transpose::Yes ? A.cols() : A.rows());
  const auto K = (trans == Transpose::Yes ? A.rows() : A.cols());

  assert(out.rows() == N && out.cols() == N);
  detail::BlasDispatch<T>::syrk(CblasRowMajor, ul, op, N, K, alpha, A.data(),
                                A.ld(), beta, out.data(), out.ld());
}

// Solves for X in-place inside `B`, overwriting it:
//   side == Left:  op(A) * X = alpha * B   (A is M x M, B is M x N)
//   side == Right: X * op(A) = alpha * B   (A is N x N, B is M x N)
// `diag == Unit` tells BLAS to treat A's diagonal as all-1s and ignore
// whatever is actually stored there.
template <class T>
void trsm(Side side, Triangular tri, Transpose trans, Diagonal diag, T alpha,
          MatrixView<const T> A, MatrixView<T> B) {
  const auto sd = (side == Side::Left ? CblasLeft : CblasRight);
  const auto ul = (tri == Triangular::Upper ? CblasUpper : CblasLower);
  const auto op = (trans == Transpose::Yes ? CblasTrans : CblasNoTrans);
  const auto dg = (diag == Diagonal::Unit ? CblasUnit : CblasNonUnit);

  const auto M = B.rows(), N = B.cols();
  const int expA = (side == Side::Left ? M : N);

  assert(A.rows() == expA && A.cols() == expA);

  detail::BlasDispatch<T>::trsm(CblasRowMajor, sd, ul, op, dg, M, N, alpha,
                                A.data(), A.ld(), B.data(), B.ld());
}

} // namespace ml::linalg
