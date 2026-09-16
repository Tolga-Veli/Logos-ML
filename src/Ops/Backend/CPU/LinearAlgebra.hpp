#pragma once

#include "Ops/Backend/CPU/BLAS.hpp"
#include "Ops/Utils.hpp"
#include "Ops/Views/Views.hpp"

namespace ml::backend::cpu {
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
void gemv(Transpose transA, T alpha, MatrixView<const T> A, VectorView<const T> x, T beta, VectorView<T> y) {
  CORE_ASSERT(x.size() == (transA == Transpose::Yes ? A.rows() : A.cols()) &&
                  y.size() == (transA == Transpose::Yes ? A.rows() : A.cols()),
              "Invalid shape");

  const auto opA = (transA == Transpose::Yes ? CblasTrans : CblasNoTrans);
  detail::BlasDispatch<T>::gemv(CblasRowMajor, opA, A.rows(), A.cols(), alpha, A.data(), A.col_stride(), x.data(),
                                x.stride(), beta, y.data(), y.stride());
}

// adds vec to every row of mat in-place
template <class T> void add_rowwise_vector(MatrixView<T> mat, VectorView<const T> vec) {
  CORE_ASSERT(mat.cols() == vec.size(), "Shape mismatch");
  for (std::size_t i = 0; i < mat.rows(); i++)
    detail::BlasDispatch<T>::axpy(mat.cols(), T{1}, vec.data(), vec.stride(), mat.data() + i * mat.col_stride(), 1);
}

// sums all rows of mat into vec in-place
template <class T> void sum_rows(MatrixView<const T> mat, VectorView<T> vec) {
  CORE_ASSERT(mat.cols() == vec.size(), "Shape mismatch");
  for (std::size_t i = 0; i < mat.rows(); i++)
    detail::BlasDispatch<T>::axpy(mat.cols(), T{1}, mat.data() + i * mat.col_stride(), 1, vec.data(), vec.stride());
}

// y := x   (plain copy, BLAS-optimized -- std::copy_n is an equally valid
// choice if x/y are both contiguous; this version also handles strided
// VectorViews with stride() != 1, which std::copy_n alone would not)
template <class T> void copy(VectorView<const T> x, VectorView<T> y) {
  CORE_ASSERT(x.size() == y.size(), "Shape mismatch");
  detail::BlasDispatch<T>::copy(x.size(), x.data(), x.stride(), y.data(), y.stride());
}

// Returns the inner product x . y
template <class T> void dot(VectorView<const T> x, VectorView<const T> y, ScalarView<T> out) {
  CORE_ASSERT(x.size() == y.size(), "Shape mismatch");
  out.value() = detail::BlasDispatch<T>::dot(x.size(), x.data(), x.stride(), y.data(), y.stride());
}

// Returns the Euclidean (L2) norm of x, i.e. sqrt(sum(x_i^2))
template <class T> void nrm2(VectorView<const T> x, ScalarView<T> out) {
  out.value() = detail::BlasDispatch<T>::nrm2(x.size(), x.data(), x.stride());
}

// Returns the sum of absolute values, sum(|x_i|)  (the L1 norm)
template <class T> void asum(VectorView<const T> x, ScalarView<T> out) {
  out.value() = detail::BlasDispatch<T>::asum(x.size(), x.data(), x.stride());
}

// Returns x . y accumulated internally in double precision, even though
// x and y are float vectors. Meaningfully more accurate than dot<float>
// for long vectors, since plain float accumulation loses precision as
// the running sum grows relative to each incoming term.
//
// Only valid for T=float -- there's no extra precision tier above double
// to accumulate into, so this is a compile error for T=double rather
// than silently degrading to plain dot().
template <class T> void dot_precise(VectorView<const T>, VectorView<const T>, ScalarView<double>) { UNREACHABLE(); }

template <>
inline void dot_precise<float>(VectorView<const float> x, VectorView<const float> y, ScalarView<double> out) {
  CORE_ASSERT(x.size() == y.size(), "Shape mismatch");
  out.value() = detail::BlasDispatch<float>::dsdot(x.size(), x.data(), x.stride(), y.data(), y.stride());
}
} // namespace ml::backend::cpu
