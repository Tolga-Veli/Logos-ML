#pragma once

#include "Core/Assert.hpp"
#include "Ops/Kernels/CPU/CrossEntropyLoss.hpp"
#include "Ops/Views/MatrixView.hpp"
#include "Ops/Views/ScalarView.hpp"
#include "Ops/Views/VectorView.hpp"
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
inline void cross_entropy(const core::Tensor &logits,
                          const core::Tensor &labels, core::Tensor &probs,
                          core::Tensor &loss) {
  CORE_VERIFY(logits.rank() == 2, "Requires logits with shape [B, C]");

  CORE_VERIFY(labels.rank() == 1, "Requires labels with shape [B]");

  CORE_VERIFY(probs.rank() == 2, "Requires probabilities with shape [B, C]");

  CORE_VERIFY(loss.rank() == 0, "Requires loss to be a scalar");

  [[maybe_unused]] const int batch = logits.shape()[0],
                             classes = logits.shape()[1];

  CORE_VERIFY(batch > 0, "Requires a non-empty batch");

  CORE_VERIFY(classes > 0, "Requires at least one class");

  CORE_VERIFY(labels.shape()[0] == batch, "Labels must have shape [B]");

  CORE_VERIFY(probs.shape()[0] == batch && probs.shape()[1] == classes,
              "Probabilities must have shape [B, C]");

  CORE_VERIFY(
      logits.device() == probs.device() && logits.device() == loss.device() &&
          logits.device() == labels.device(),
      "Logits, labels, probabilities, and loss must be on the same device");

  CORE_VERIFY(logits.dtype() == probs.dtype() && logits.dtype() == loss.dtype(),
              "Logits, probabilities, and loss must have the same dtype");

  CORE_VERIFY(labels.dtype() == core::DType::Int32,
              "Labels must have dtype int32");
  CORE_VERIFY(logits.dtype() == core::DType::Float32 ||
                  logits.dtype() == core::DType::Float64,
              "Cross entropy requires floating-point logits");

  using kernels::MatrixView;
  using kernels::ScalarView;
  using kernels::VectorView;

  kernels::dispatch(
      logits.device(), logits.dtype(), [&]<memory::DeviceType D, class T>() {
        if constexpr (D == memory::DeviceType::CPU)
          kernels::cpu::cross_entropy(
              MatrixView<const T>(logits), VectorView<const int>(labels),
              MatrixView<T>(probs), ScalarView<T>(loss));
        else {
          CORE_VERIFY(false, "not implemented yet");
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
inline void cross_entropy_backward(const core::Tensor &probs,
                                   const core::Tensor &labels,
                                   core::Tensor &grad) {
  CORE_VERIFY(probs.rank() == 2, "Requires probabilities with shape [B, C]");

  CORE_VERIFY(labels.rank() == 1, "Requires labels with shape [B]");

  CORE_VERIFY(grad.rank() == 2, "Requires gradient with shape [B, C]");

  [[maybe_unused]] const int batch = probs.shape()[0],
                             classes = probs.shape()[1];

  CORE_VERIFY(batch > 0, "Requires a non-empty batch");

  CORE_VERIFY(classes > 0, "Requires at least one class");

  CORE_VERIFY(labels.shape()[0] == batch, "Labels must have shape [B]");

  CORE_VERIFY(grad.shape()[0] == batch && grad.shape()[1] == classes,
              "Gradient must have shape [B, C]");

  CORE_VERIFY(probs.device() == labels.device() &&
                  probs.device() == grad.device(),
              "Probabilities, labels, and "
              "gradient must be on the same "
              "device");

  CORE_VERIFY(probs.dtype() == grad.dtype(), "Probabilities and gradient "
                                             "must have the same dtype");

  CORE_VERIFY(labels.dtype() == core::DType::Int32,
              "Labels must have dtype int32");
  CORE_VERIFY(probs.dtype() == core::DType::Float32 ||
                  probs.dtype() == core::DType::Float64,
              "Cross entropy backward requires floating-point probabilities");

  kernels::dispatch(
      probs.device(), probs.dtype(), [&]<memory::DeviceType D, class T>() {
        if constexpr (D == memory::DeviceType::CPU) {
          kernels::cpu::cross_entropy_backward(MatrixView<const T>(probs),
                                               VectorView<const int>(labels),
                                               MatrixView<T>(grad));
        } else {
          CORE_VERIFY(false, "Cross entropy backward is "
                             "not implemented for this "
                             "device");
        }
      });
}
} // namespace ml::ops
