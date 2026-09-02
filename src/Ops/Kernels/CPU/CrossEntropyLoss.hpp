#pragma once

#include "Core/Assert.hpp"

#include "Ops/Views/MatrixView.hpp"
#include "Ops/Views/ScalarView.hpp"
#include "Ops/Views/VectorView.hpp"

#include "Softmax.hpp"

#include <cmath>

namespace ml::kernels::cpu {

template <class T>
inline void cross_entropy(MatrixView<const T> logits,
                          VectorView<const int> labels, MatrixView<T> probs,
                          ScalarView<T> loss) {
  const int batch = logits.rows(), classes = logits.cols();

  softmax(logits, probs);

  const int *labels_data = labels.data();
  const T *logit_data = logits.data();
  T total{0};
  for (int i = 0; i < batch; i++) {
    const int label = labels_data[i];

    CORE_VERIFY(label >= 0 && label < classes, "Label is outside class range");

    // Do not compute -log(softmax(logits)[label]) from `probs`: a valid
    // probability can underflow to zero.  The equivalent log-sum-exp form
    // stays finite after shifting by the row maximum.  See Blanchard,
    // Higham & Higham (2021), doi.org/10.1093/imanum/draa038.
    const T *row = logit_data + i * classes;
    const T max_logit = *std::max_element(row, row + classes);
    T exp_sum{0};
    for (int j = 0; j < classes; ++j)
      exp_sum += std::exp(row[j] - max_logit);
    total += std::log(exp_sum) + max_logit - row[label];
  }

  loss.value() = total / static_cast<T>(batch);
}

template <class T>
inline void cross_entropy_backward(MatrixView<const T> probs,
                                   VectorView<const int> labels,
                                   MatrixView<T> grad) {
  const int batch = probs.rows(), classes = probs.cols();

  const T inv_batch = T{1} / static_cast<T>(batch);
  const T *probs_data = probs.data();
  const int *labels_data = labels.data();
  T *grad_data = grad.data();

  for (int i = 0; i < batch; i++) {
    const int offset = i * classes, label = labels_data[i];

    CORE_VERIFY(label >= 0 && label < classes, "Label is outside class range");

    for (int j = 0; j < classes; j++)
      grad_data[offset + j] =
          (probs_data[offset + j] - (j == label ? T{1} : T{0})) * inv_batch;
  }
}
} // namespace ml::kernels::cpu
