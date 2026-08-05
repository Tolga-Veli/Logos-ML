#pragma once

#include "BlasOps.hpp"

namespace ml::linalg {

// y := alpha * op(A) * x + beta * y
//
// A is stored as (rows x cols) regardless of transA -- BLAS always wants
// A's *storage* dimensions, not op(A)'s logical dimensions. Only the
// expected lengths of x and y flip based on transA.
//
//   transA == No:  op(A) is A,   shape (rows x cols) -> y:rows, x:cols
//   transA == Yes: op(A) is A^T, shape (cols x rows) -> y:cols, x:rows
template <class T>
void gemv_impl(Transpose transA, T alpha, MatrixView<const T> A,
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

// adds vec to every row of mat in-place
template <class T>
void add_row_vector_impl(MatrixView<T> mat, VectorView<const T> vec) {
  assert(mat.cols() == vec.size());

  for (int i = 0; i < mat.rows(); i++)
    detail::BlasDispatch<T>::axpy(mat.cols(), T{1}, vec.data(), vec.inc(),
                                  mat.data() + i * mat.ld(), 1);
}

template <class T> void add_row_vector(Tensor<T> &mat, const Tensor<T> &vec) {
  assert(mat.rank() == 2 && vec.rank() == 1);
  add_row_vector_impl(core::as_matrix(mat), core::as_vector(vec));
}

// sums all rows of mat into vec in-place
template <class T>
void sum_rows_impl(MatrixView<const T> mat, VectorView<T> vec) {
  assert(mat.cols() == vec.size());

  for (int i = 0; i < mat.rows(); i++)
    detail::BlasDispatch<T>::axpy(mat.cols(), T{1}, mat.data() + i * mat.ld(),
                                  1, vec.data(), vec.inc());
}

template <class T> void sum_rows(const Tensor<T> &mat, Tensor<T> &vec) {
  assert(mat.rank() == 2 && vec.rank() == 1);
  sum_rows_impl(core::as_matrix(mat), core::as_vector(vec));
}

} // namespace ml::linalg
