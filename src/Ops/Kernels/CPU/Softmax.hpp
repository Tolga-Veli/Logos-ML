#pragma once

#include "Core/Tensor.hpp"

#include <algorithm>
#include <cmath>

namespace ml::kernels::cpu {
using core::Tensor;

template <class T> void softmax(const Tensor &input, Tensor &output) {
  CORE_VERIFY(input.rank() == 2, "Invalid tensor argument");

  const auto batch = input.shape()[0], classes = input.shape()[1];
  for (int i = 0; i < batch; i++) {
    const T *in = input.data<T>() + i * classes;
    T *out = output.data<T>() + i * classes;

    const T maxv = *std::max_element(in, in + classes);
    T sum{0};

    for (int j = 0; j < classes; j++) {
      out[j] = std::exp(in[j] - maxv);
      sum += out[j];
    }

    CORE_VERIFY(sum > T{0}, "Trying to divide by 0");

    for (int j = 0; j < classes; j++)
      out[j] /= sum;
  }
}
} // namespace ml::kernels::cpu
