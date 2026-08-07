#pragma once

#include "Core/Assert.hpp"
#include "Softmax.hpp"

#include <cmath>

namespace ml::ops {
// return loss and argument parameter is a probability distribution on the
// possible outputs
// probs tensor must be the same shape as logits

template <class T>
inline T cross_entropy(const Tensor &logits, const Tensor &labels,
                       Tensor &probs) {
  CORE_VERIFY(logits.rank() == 2, "Expects a matrix");
  CORE_VERIFY(labels.rank() == 1, "Expects a row");

  const auto batch = logits.shape()[0], classes = logits.shape()[1];
  CORE_VERIFY(labels.shape()[0] == batch,
              "Batch size must be equal to the batch of labels");

  softmax(logits, probs);

  // loss = -mean(log(prob[correct_class]))
  T total{0};
  const int *label_data = labels.data<int>();
  auto *prob_data = probs.data<T>();
  for (int i = 0; i < batch; i++) {
    const int label = label_data[i];
    const T prob = prob_data[i * classes + label];
    total -= std::log(prob + T{1e-7});
  }

  return total / static_cast<T>(batch);
}

// Gradient of softmax + cross-entropy w.r.t. logits
// grad[i][j] = (probs[i][j] - 1) / batch if i == label
//            = (probs[i][j]) / batch otherwise

// grad must be the same shape as probs
template <class T>
inline void cross_entropy_backward(const Tensor &probs, const Tensor &labels,
                                   Tensor &grad) {
  const auto batch = probs.shape()[0], classes = probs.shape()[1];
  auto *grad_data = grad.data<T>();
  std::copy_n(probs.data<T>(), probs.num_elements(), grad_data);

  for (int i = 0; i < batch; i++)
    grad_data[i * classes + labels.data<int>()[i]] -= T{1};

  // normalize by batch size
  const T inv_batch = T{1} / static_cast<T>(batch);
  for (int i = 0; i < grad.num_elements(); i++)
    grad_data[i] *= inv_batch;
}
} // namespace ml::ops
