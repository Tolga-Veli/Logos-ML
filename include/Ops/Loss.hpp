#pragma once

#include "Core/Assert.hpp"
#include "Ops/Backend/CPU/CrossEntropyLoss.hpp"
#include "Ops/Backend/Dispatch.hpp"
#include "Ops/Validation.hpp"
#include "Ops/Views/Views.hpp"

namespace ml::ops {

/*
 * @brief Computes the mean categorical cross-entropy loss
 *
 * @note B = batches, C = classes
 *
 * Given logits with shape [B, C] and an integer class labels with
 * shape [B], first computes:
 *
 *   probs = softmax(logits)
 *
 * and then:
 *   loss = -(1 / batches) * sum_i Log(probs[i][labels_i])
 *
 * @param logits Logits with shape [B, C]
 * @param labels Labels with shape [B]
 * @param probs Probabilities with shape [B, C]
 * @param loss Mean loss as a scalar
 */
inline void cross_entropy(const core::Tensor &logits, const core::Tensor &labels, core::Tensor &probs,
                          core::Tensor &loss) {
  require_same_device(logits, labels, probs, loss);
  require_same_dtype(logits, probs, loss);

  dispatch(logits.device(), logits.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::cross_entropy<T>(logits, labels, probs, loss);
    else
      UNREACHABLE();
  });
}

/*
 * @brief Computes the gradient of softmax followed by cross-entropy
 *
 * @note B = batches, C = classes
 *
 * Given Probabilities with shape [B, C] and labels with shape
 * [B], computes the gradient with respect to the logits:
 *
 *   grad[i][j] = (probs[i][j] - 1) / batches, if j == labels[i]
 *                 probs(i)[j] / batches,      otherwise
 *
 * @param probs Probabilities with shape [B, C].
 * @param labels Labels with shape [B].
 * @param grad Output gradient with shape [B, C].
 */
inline void cross_entropy_backward(const core::Tensor &probs, const core::Tensor &labels, core::Tensor &grad) {
  require_same_device(probs, labels, grad);
  require_same_dtype(probs, grad);

  dispatch(probs.device(), probs.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::cross_entropy_backward<T>(probs, labels, grad);
    else
      UNREACHABLE();
  });
}
} // namespace ml::ops
