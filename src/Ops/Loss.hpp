#pragma once

#include "Core/Assert.hpp"
#include "Ops/Kernels/CPU/CrossEntropyLoss.hpp"
#include "Ops/Views/MatrixView.hpp"
#include "Ops/Views/ScalarView.hpp"
#include "Ops/Views/VectorView.hpp"
#include "Softmax.hpp"

#include <cmath>

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
inline void cross_entropy(const Tensor &logits, const Tensor &labels,
                          Tensor &probs, Tensor &loss) {
  CORE_ASSERT(logits.rank() == 2, "Requires logits with shape [B, C]");

  CORE_ASSERT(labels.rank() == 1, "Requires labels with shape [B]");

  CORE_ASSERT(probs.rank() == 2, "Requires probabilities with shape [B, C]");

  CORE_ASSERT(loss.rank() == 0, "Requires loss to be a scalar");

  [[maybe_unused]] const int batch = logits.shape()[0],
                             classes = logits.shape()[1];

  CORE_ASSERT(labels.shape()[0] == batch, "Labels must have shape [B]");

  CORE_ASSERT(probs.shape()[0] == batch && probs.shape()[1] == classes,
              "Probabilities must have shape [B, C]");

  CORE_ASSERT(
      logits.device() == probs.device() && logits.device() == loss.device() &&
          logits.device() == labels.device(),
      "Logits, labels, probabilities, and loss must be on the same device");

  CORE_ASSERT(logits.dtype() == probs.dtype() && logits.dtype() == loss.dtype(),
              "Logits, probabilities, and loss must have the same dtype");

  CORE_ASSERT(labels.dtype() == core::DType::Int32,
              "Labels must have dtype int32");

  using kernels::MatrixView;
  using kernels::ScalarView;
  using kernels::TensorView;
  using kernels::VectorView;

  kernels::dispatch(
      logits.device(), logits.dtype(), [&]<memory::DeviceType D, class T>() {
        if constexpr (D == memory::DeviceType::CPU)
          kernels::cpu::cross_entropy(
              MatrixView<const T>(logits), VectorView<const int>(labels),
              MatrixView<T>(probs), ScalarView<T>(loss));
        else {
          CORE_ASSERT(false, "not implemented yet");
        }
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
inline void cross_entropy_backward(const Tensor &probs, const Tensor &labels,
                                   Tensor &grad) {
  CORE_ASSERT(probs.rank() == 2, "Requires probabilities with shape [B, C]");

  CORE_ASSERT(labels.rank() == 1, "Requires labels with shape [B]");

  CORE_ASSERT(grad.rank() == 2, "Requires gradient with shape [B, C]");

  [[maybe_unused]] const int batch = probs.shape()[0],
                             classes = probs.shape()[1];

  CORE_ASSERT(batch > 0, "Requires a non-empty batch");

  CORE_ASSERT(classes > 0, "Requires at least one class");

  CORE_ASSERT(labels.shape()[0] == batch, "Labels must have shape [B]");

  CORE_ASSERT(grad.shape()[0] == batch && grad.shape()[1] == classes,
              "Gradient must have shape [B, C]");

  CORE_ASSERT(probs.device() == labels.device() &&
                  probs.device() == grad.device(),
              "Probabilities, labels, and "
              "gradient must be on the same "
              "device");

  CORE_ASSERT(probs.dtype() == grad.dtype(), "Probabilities and gradient "
                                             "must have the same dtype");

  CORE_ASSERT(labels.dtype() == core::DType::Int32,
              "Labels must have dtype int32");

  kernels::dispatch(
      probs.device(), probs.dtype(), [&]<memory::DeviceType D, class T>() {
        if constexpr (D == memory::DeviceType::CPU) {
          kernels::cpu::cross_entropy_backward(MatrixView<const T>(probs),
                                               VectorView<const int>(labels),
                                               MatrixView<T>(grad));
        } else {
          CORE_ASSERT(false, "Cross entropy backward is "
                             "not implemented for this "
                             "device");
        }
      });
}
} // namespace ml::ops
