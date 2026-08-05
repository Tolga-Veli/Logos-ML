#pragma once

#include "Core/Tensor.hpp"

#include <algorithm>
#include <cassert>

namespace ml::ops {
using core::Tensor;

// out[i] = max(x[i], 0)
// Writes into a freshly allocated output tensor

template <class T> Tensor<T> relu(const Tensor<T> &x) {
  assert(x.is_contiguous());

  Tensor<T> out(x.shape());
  const T *src = x.data();
  T *dst = out.data();
  const int n = x.num_elements();

  for (int i = 0; i < n; i++)
    dst[i] = std::max(src[i], 0);

  return out;
}

// grad_input[i] = grad_out[i] if x[i] > 0, else 0
template <class T>
Tensor<T> relu_backward(const Tensor<T> &grad_out, const Tensor<T> &x) {
  assert(grad_out.is_contiguous());
  assert(x.is_contiguous());
  assert(grad_out.num_elements() == x.num_elements());

  Tensor<T> grad_in(x.shape());
  const T *g = grad_out.data(), src = x.data();
  T *dst = grad_in.data();
  const int n = x.num_elements();

  for (int i = 0; i < n; i++)
    dst[i] = src[i] > T{0} ? g[i] : T{0};

  return grad_in;
}
} // namespace ml::ops
