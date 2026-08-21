#pragma once

#include "Ops/Views/MatrixView.hpp"

#include <algorithm>
#include <cmath>

namespace ml::kernels::cpu {

template <class T>
inline void softmax(MatrixView<const T> input, MatrixView<T> output) {
  const int batch = input.rows(), classes = input.cols();

  const T *input_data = input.data();
  T *output_data = output.data();
  for (int i = 0; i < batch; i++) {
    const int offset = i * classes;

    const T *in = input_data + offset;
    T *out = output_data + offset;

    // Find the maximum logit:
    //
    // max = max(x_0, ..., x_{C-1})
    //
    // Subtracting it before exp() prevents large positive logits from
    // overflowing

    const T maxv = *std::max_element(in, in + classes);

    T sum{0};

    // exp(x_j - max)
    for (int j = 0; j < classes; j++) {
      const T value = std::exp(in[j] - maxv);

      CORE_ASSERT(std::isfinite(value), "Produced a non-finite value");

      out[j] = value;
      sum += value;
    }

    CORE_ASSERT(std::isfinite(sum) && sum > T{0},
                "Normalization sum must be finite and positive");

    // Normalize:
    //
    // P_j = exp(x_j - max) / sum

    const T inv_sum = T{1} / sum;
    for (int j = 0; j < classes; j++)
      out[j] *= inv_sum;
  }
}
} // namespace ml::kernels::cpu
