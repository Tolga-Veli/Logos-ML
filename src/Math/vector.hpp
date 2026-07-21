#pragma once

#include "BlasOps.hpp"
#include "Core/VectorView.hpp"

namespace ml::linalg {
template <class T> using VectorView = core::VectorView<T>;

// x := alpha * x   (in place)
template <class T> void scal(T alpha, VectorView<T> x) {
  detail::BlasDispatch<T>::scal(x.size(), alpha, x.data(), x.inc());
}

// y := alpha * x + y   (in place, y is both input and output)
template <class T> void axpy(T alpha, VectorView<const T> x, VectorView<T> y) {
  assert(x.size() == y.size());

  detail::BlasDispatch<T>::axpy(x.size(), alpha, x.data(), x.inc(), y.data(),
                                y.inc());
}

// y := x   (plain copy, BLAS-optimized -- std::copy_n is an equally valid
// choice if x/y are both contiguous; this version also handles strided
// VectorViews with inc() != 1, which std::copy_n alone would not)
template <class T> void copy(VectorView<const T> x, VectorView<T> y) {
  assert(x.size() == y.size());

  detail::BlasDispatch<T>::copy(x.size(), x.data(), x.inc(), y.data(), y.inc());
}

// Returns the inner product x . y
template <class T>
[[nodiscard]] T dot(VectorView<const T> x, VectorView<const T> y) {
  assert(x.size() == y.size());

  return detail::BlasDispatch<T>::dot(x.size(), x.data(), x.inc(), y.data(),
                                      y.inc());
}

// Returns the Euclidean (L2) norm of x, i.e. sqrt(sum(x_i^2))
template <class T> [[nodiscard]] T nrm2(VectorView<const T> x) {
  return detail::BlasDispatch<T>::nrm2(x.size(), x.data(), x.inc());
}

// Returns the sum of absolute values, sum(|x_i|)  (the L1 norm)
template <class T> [[nodiscard]] T asum(VectorView<const T> x) {
  return detail::BlasDispatch<T>::asum(x.size(), x.data(), x.inc());
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
[[nodiscard]] double dot_precise(VectorView<const T>, VectorView<const T>) {
  static_assert(std::is_same_v<T, float>,
                "dot_precise is only meaningful for float vectors; "
                "double already has no higher-precision tier to "
                "accumulate into, use dot<double> instead.");
}

template <>
[[nodiscard]] inline double dot_precise<float>(VectorView<const float> x,
                                               VectorView<const float> y) {
  assert(x.size() == y.size());
  return detail::dsdot(x.size(), x.data(), x.inc(), y.data(), y.inc());
}
} // namespace ml::linalg
