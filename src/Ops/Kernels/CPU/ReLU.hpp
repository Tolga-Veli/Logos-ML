#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"

#include <algorithm>
#include <cmath>

namespace ml::kernels::cpu {
using core::Tensor;

// out[i] = max(x[i], 0)
// x and out must be same shape and x must be contiguous
template <class T> void relu(const Tensor &x, Tensor &out) {
  CORE_VERIFY(x.is_contiguous(), "x must be contiguous");

  const T *src = x.data<T>();
  T *dst = out.data<T>();
  const auto n = x.num_elements();

  for (int i = 0; i < n; i++)
    dst[i] = std::max(src[i], T{0});
}

// grad_input[i] = grad_out[i] if x[i] > 0, else 0
// grad_in and grad_out must be same shape
// grad_out and x must be contiguous
template <class T>
void relu_backward(const Tensor &grad_out, const Tensor &x, Tensor &grad_in) {
  CORE_VERIFY(grad_out.is_contiguous(), "grad_out must be contiguous");
  CORE_VERIFY(x.is_contiguous(), "x must be contiguous");
  CORE_VERIFY(grad_out.num_elements() == x.num_elements(),
              "grad_out.numel() must be equal to x.numel()");

  const T *g = grad_out.data<T>(), *src = x.data<T>();
  T *dst = grad_in.data<T>();
  const auto n = x.num_elements();

  for (int i = 0; i < n; i++)
    dst[i] = src[i] > T{0} ? g[i] : T{0};
}
} // namespace ml::kernels::cpu
