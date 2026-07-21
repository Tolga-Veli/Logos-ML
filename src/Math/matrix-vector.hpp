#pragma once

#include "BlasOps.hpp"
#include "Core/MatrixView.hpp"
#include "Core/VectorView.hpp"

namespace ml::linalg {
template <class T> using VectorView = core::VectorView<T>;
template <class T> using MatrixView = core::MatrixView<T>;

// y := alpha * op(A) * x + beta * y
//
// A is stored as (rows x cols) regardless of transA -- BLAS always wants
// A's *storage* dimensions, not op(A)'s logical dimensions. Only the
// expected lengths of x and y flip based on transA.
//
//   transA == No:  op(A) is A,   shape (rows x cols) -> y:rows, x:cols
//   transA == Yes: op(A) is A^T, shape (cols x rows) -> y:cols, x:rows
template <class T>
void gemv(Transpose transA, T alpha, MatrixView<const T> A,
          VectorView<const T> x, T beta, VectorView<T> y) {
  const auto opA = (transA == Transpose::Yes ? CblasTrans : CblasNoTrans);

  const auto yLen = (transA == Transpose::Yes ? A.cols() : A.rows());
  const auto xLen = (transA == Transpose::Yes ? A.rows() : A.cols());

  assert(x.size() == xLen);
  assert(y.size() == yLen);

  detail::BlasDispatch<T>::gemv(CblasRowMajor, opA, A.rows(), A.cols(), alpha,
                                A.data(), A.ld(), x.data(), x.inc(), beta,
                                y.data(), y.inc());
}

// y := alpha * A * x + beta * y
//
// A is symmetric N x N; only the `uplo` triangle of A is actually read
// (same contract as syrk/symm -- the other triangle's contents are
// ignored, not validated). No transpose parameter: op(A) == A always,
// since A^T == A for a symmetric matrix.
template <class T>
void symv(Triangular uplo, T alpha, MatrixView<const T> A,
          VectorView<const T> x, T beta, VectorView<T> y) {
  const auto ul = (uplo == Triangular::Upper ? CblasUpper : CblasLower);

  assert(A.rows() == A.cols());
  assert(x.size() == A.cols());
  assert(y.size() == A.rows());

  detail::BlasDispatch<T>::symv(CblasRowMajor, ul, A.rows(), alpha, A.data(),
                                A.ld(), x.data(), x.inc(), beta, y.data(),
                                y.inc());
}

// x := op(A) * x   (in place -- x is both input and output)
//
// A is triangular N x N; `uplo` selects which triangle holds real data,
// `diag` says whether to treat the diagonal as implicit 1s (Unit) or
// read it from A (NonUnit).
template <class T>
void trmv(Triangular uplo, Transpose trans, Diagonal diag,
          MatrixView<const T> A, VectorView<T> x) {
  const auto ul = (uplo == Triangular::Upper ? CblasUpper : CblasLower);
  const auto tr = (trans == Transpose::Yes ? CblasTrans : CblasNoTrans);
  const auto dg = (diag == Diagonal::Unit ? CblasUnit : CblasNonUnit);

  assert(A.rows() == A.cols());
  assert(x.size() == A.rows());

  detail::BlasDispatch<T>::trmv(CblasRowMajor, ul, tr, dg, A.rows(), A.data(),
                                A.ld(), x.data(), x.inc());
}

// Solves op(A) * x = b for x, overwriting x in place (pass b in via x --
// same in-place contract as trsm, just the vector-equation version).
template <class T>
void trsv(Triangular uplo, Transpose trans, Diagonal diag,
          MatrixView<const T> A, VectorView<T> x) {
  const auto ul = (uplo == Triangular::Upper ? CblasUpper : CblasLower);
  const auto tr = (trans == Transpose::Yes ? CblasTrans : CblasNoTrans);
  const auto dg = (diag == Diagonal::Unit ? CblasUnit : CblasNonUnit);

  assert(A.rows() == A.cols());
  assert(x.size() == A.rows());

  detail::BlasDispatch<T>::trsv(CblasRowMajor, ul, tr, dg, A.rows(), A.data(),
                                A.ld(), x.data(), x.inc());
}

// A := alpha * x * y^T + A   (rank-1 update, in place)
//
// x has length A.rows(), y has length A.cols() -- note this is the one
// Level 2 op where A is *not* required to be square.
template <class T>
void ger(T alpha, VectorView<const T> x, VectorView<const T> y,
         MatrixView<T> A) {
  assert(x.size() == A.rows());
  assert(y.size() == A.cols());

  detail::BlasDispatch<T>::ger(CblasRowMajor, A.rows(), A.cols(), alpha,
                               x.data(), x.inc(), y.data(), y.inc(), A.data(),
                               A.ld());
}
} // namespace ml::linalg
