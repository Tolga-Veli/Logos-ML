#pragma once

#include "Ops/Views/MatrixView.hpp"

#include <algorithm>
#include <cmath>

namespace ml::backend::cpu {

template <class T> inline void softmax(MatrixView<const T> input, MatrixView<T> output) {
  CORE_ASSERT(input.rows() == output.rows() && input.cols() == output.cols(), "Shape mismatch");
  CORE_ASSERT(input.cols() > 0, "Requires at least one class");

  const std::size_t batch = input.rows(), classes = input.cols();
  for (std::size_t i = 0; i < batch; i++) {
    T maxv = input(i, 0);
    for (std::size_t j = 1; j < classes; j++)
      maxv = std::max(maxv, input(i, j));

    T sum{0};
    for (std::size_t j = 0; j < classes; j++) {
      const T val = std::exp(input(i, j) - maxv);

      output(i, j) = val;
      sum += val;
    }

    CORE_ASSERT(std::isfinite(sum) && sum > T{0}, "Normalization sum must be finite and positive");

    const T inv_sum = T{1} / sum;
    for (std::size_t j = 0; j < classes; j++)
      output(i, j) *= inv_sum;
  }
}
} // namespace ml::backend::cpu
