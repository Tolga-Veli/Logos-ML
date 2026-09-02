#pragma once

#ifndef BLAS_BACKEND
#include "Ops/Kernels/CPU/Backend/BLAS.hpp"
#endif

#include "Ops/Utils.hpp"
#include "Ops/Views/MatrixView.hpp"
#include "Ops/Views/ScalarView.hpp"
#include "Ops/Views/VectorView.hpp"

namespace ml::kernels::cpu {
using ops::Transpose;

// y := alpha * op(A) * x + beta * y
//
// A is stored as (rows x cols) regardless of transA -- BLAS always wants
// A's *storage* dimensions, not op(A)'s logical dimensions. Only the
// expected lengths of x and y flip based on transA.
//
//   transA == No:  op(A) is A,   shape (rows x cols) -> y:rows, x:cols
//   transA == Yes: op(A) is A^T, shape (cols x rows) -> y:cols, x:rows
template <class T>
void gemv_kernel(Transpose transA, T alpha, MatrixView<const T> A,
                 VectorView<const T> x, T beta, VectorView<T> y) {
  const auto opA = (transA == Transpose::Yes ? CblasTrans : CblasNoTrans);
  const auto yLen = (transA == Transpose::Yes ? A.cols() : A.rows());
  const auto xLen = (transA == Transpose::Yes ? A.rows() : A.cols());

  CORE_VERIFY(x.size() == xLen, "Invalid shape");
  CORE_VERIFY(y.size() == yLen, "Invalid shape");

  detail::BlasDispatch<T>::gemv(CblasRowMajor, opA, A.rows(), A.cols(), alpha,
                                A.data(), A.ld(), x.data(), x.inc(), beta,
                                y.data(), y.inc());
}

// adds vec to every row of mat in-place
template <class T>
void add_rowwise_vector_kernel(MatrixView<T> mat, VectorView<const T> vec) {
  CORE_VERIFY(mat.cols() == vec.size(), "Invalid shape");

  for (int i = 0; i < mat.rows(); i++)
    detail::BlasDispatch<T>::axpy(mat.cols(), T{1}, vec.data(), vec.inc(),
                                  mat.data() + i * mat.ld(), 1);
}

// sums all rows of mat into vec in-place
template <class T>
void sum_rows_kernel(MatrixView<const T> mat, VectorView<T> vec) {
  CORE_VERIFY(mat.cols() == vec.size(), "Invalid shape");
  for (int i = 0; i < mat.rows(); i++)
    detail::BlasDispatch<T>::axpy(mat.cols(), T{1}, mat.data() + i * mat.ld(),
                                  1, vec.data(), vec.inc());
}

// y := x   (plain copy, BLAS-optimized -- std::copy_n is an equally valid
// choice if x/y are both contiguous; this version also handles strided
// VectorViews with inc() != 1, which std::copy_n alone would not)
template <class T> void copy(VectorView<const T> x, VectorView<T> y) {
  CORE_VERIFY(x.size() == y.size(), "x and y must be of same size");
  detail::BlasDispatch<T>::copy(x.size(), x.data(), x.inc(), y.data(), y.inc());
}

// Returns the inner product x . y
template <class T>
void dot(VectorView<const T> x, VectorView<const T> y, ScalarView<T> out) {
  CORE_VERIFY(x.size() == y.size(), "x and y must be of same size");
  out.value() = detail::BlasDispatch<T>::dot(x.size(), x.data(), x.inc(),
                                             y.data(), y.inc());
}

// Returns the Euclidean (L2) norm of x, i.e. sqrt(sum(x_i^2))
template <class T> void nrm2(VectorView<const T> x, ScalarView<T> out) {
  out.value() = detail::BlasDispatch<T>::nrm2(x.size(), x.data(), x.inc());
}

// Returns the sum of absolute values, sum(|x_i|)  (the L1 norm)
template <class T> void asum(VectorView<const T> x, ScalarView<T> out) {
  out.value() = detail::BlasDispatch<T>::asum(x.size(), x.data(), x.inc());
}

// Returns x . y accumulated internally in double precision, even though
// x and y are float vectors. Meaningfully more accurate than dot<float>
// for long vectors, since plain float accumulation loses precision as
// the running sum grows relative to each incoming term.
//
// Only valid for T=float -- there's no extra precision tier above double
// to accumulate into, so this is a compile error for T=double rather
// than silently degrading to plain dot().
template <class T>
void dot_precise(VectorView<const T>, VectorView<const T>, ScalarView<double>) {
  CORE_VERIFY(false, "dot_precise only supports float input tensors");
}

template <>
inline void dot_precise<float>(VectorView<const float> x,
                               VectorView<const float> y,
                               ScalarView<double> out) {
  CORE_VERIFY(x.size() == y.size(), "x and y must be of same size");
  out.value() = detail::BlasDispatch<float>::dsdot(x.size(), x.data(), x.inc(),
                                                   y.data(), y.inc());
}
} // namespace ml::kernels::cpu
