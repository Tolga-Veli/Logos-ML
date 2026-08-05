#pragma once

#include "Core/Tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace ml::ops {
using core::Tensor;
// input - [batch, classes]
// output - [batch, classes]
template <class T> Tensor<T> softmax(const Tensor<T> &input) {
  assert(input.rank() == 2);

  const auto batch = input.shape()[0];
  const auto classes = input.shape()[1];

  Tensor<T> output(input.shape());

  for (int i = 0; i < batch; i++) {
    const T *in = input.data() + i * classes;
    T *out = output.data() + i * classes;

    const T maxv = *std::max_element(in, in + classes);
    T sum{0};

    for (int j = 0; j < classes; j++) {
      out[j] = std::exp(in[j] - maxv);
      sum += out[j];
    }

    assert(sum > T{0});

    for (int j = 0; j < classes; j++)
      out[j] /= sum;
  }

  return output;
}
} // namespace ml::ops
