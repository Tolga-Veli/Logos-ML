#pragma once

#include "Core/Assert.hpp"
#include "Ops/Views/Views.hpp"

#include "Softmax.hpp"

#include <cmath>

namespace ml::backend::cpu {

template <class T> inline void cross_entropy(MatrixView<const T> logits, VectorView<const int> labels,
                                             MatrixView<T> probs, ScalarView<T> loss) {
  CORE_ASSERT(logits.rows() == probs.rows(), "Shape mismatch");

  softmax(logits, probs);

  const auto batch = logits.rows(), classes = logits.cols();
  T sum{0};
  for (std::size_t i = 0; i < batch; i++) {
    const auto label = labels[i];

    CORE_VERIFY(label >= 0 && label < classes, "Label is outside class range");

    // Do not compute -log(softmax(logits)[label]) from probs: a valid
    // probability can underflow to zero.  The equivalent log-sum-exp form
    // stays finite after shifting by the row maximum.
    T maxv = logits(i, 0);
    for (std::size_t j = 1; j < classes; j++)
      maxv = std::max(maxv, logits(i, j));

    T exp_sum{0};
    for (std::size_t j = 0; j < classes; j++) {
      const T val = std::exp(logits(i, j) - maxv);
      CORE_ASSERT(std::isfinite(val), "NaN");
      probs(i, j) = val;
      exp_sum += val;
    }

    sum += (maxv - logits(i, label)) + std::log(exp_sum);

    const T inv_sum = T{1} / exp_sum;
    for (std::size_t j = 0; j < classes; j++)
      probs(i, j) *= inv_sum;
  }

  loss.value() = sum / static_cast<T>(batch);
}

template <class T>
inline void cross_entropy_backward(MatrixView<const T> probs, VectorView<const int> labels, MatrixView<T> grad) {
  const auto batch = probs.rows(), classes = probs.cols();
  const T inv_batch = T{1} / static_cast<T>(batch);
  for (std::size_t i = 0; i < batch; i++) {
    const auto label = labels[i];

    CORE_VERIFY(label >= 0 && label < classes, "Label is outside class range");

    for (std::size_t j = 0; j < classes; j++)
      grad(i, j) = (probs(i, j) - (j == label ? T{1} : T{0})) * inv_batch;
  }
}
} // namespace ml::backend::cpu
