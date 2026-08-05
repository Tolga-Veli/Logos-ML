#pragma once

#include "Core/Tensor.hpp"
#include "softmax.hpp"

#include <cassert>
#include <cmath>

namespace ml::ops {
template <class T>
std::pair<T, Tensor<T>> cross_entropy(const Tensor<T> &logits,
                                      const Tensor<int> &labels) {
  assert(logits.rank() == 2);
  assert(labels.rank() == 1);

  const auto batch = logits.shape()[0], classes = logits.shape()[1];
  assert(labels.shape()[0] == batch);

  Tensor<T> probs = softmax(logits);

  // loss = -mean(log(prob[correct_class]))
  T total{0};
  auto *label_data = labels.data();
  auto *prob_data = probs.data();
  for (int i = 0; i < batch; i++) {
    const auto label = label_data[i];
    const T prob = prob_data[i * classes + label];
    total -= std::log(prob + T{1e-7});
  }

  return {total / static_cast<T>(batch), probs};
}

// Gradient of softmax + cross-entropy w.r.t. logits
// grad[i][j] = (probs[i][j] - 1) / batch if i == label
//            = (probs[i][j]) / batch otherwise

template <class T>
Tensor<T> cross_entropy_backward(const Tensor<T> &probs,
                                 const Tensor<int> &labels) {
  const auto batch = probs.shape()[0], classes = probs.shape()[1];

  Tensor<T> grad(probs.shape());
  auto *grad_data = grad.data();
  std::copy_n(probs.data(), probs.num_elements(), grad_data);

  for (int i = 0; i < batch; i++)
    grad_data[i * classes + labels.data()[i]] -= T{1};

  // normalize by batch size
  const T inv_batch = T{1} / static_cast<T>(batch);
  for (int i = 0; i < grad.num_elements(); i++)
    grad_data[i] *= inv_batch;

  return grad;
}
} // namespace ml::ops
