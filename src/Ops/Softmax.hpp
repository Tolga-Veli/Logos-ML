#pragma once

#include "Core/Tensor.hpp"
#include "Kernels/CPU/Softmax.hpp"
#include "Ops/Kernels/Dispatch.hpp"

namespace ml::ops {
/*
 * @brief Computes the softmax function independently for each batch row
 *
 * @note B = batches, C = number of classes
 *
 * Given logits with shape [B, C], computes:
 *    probs[i][j] = exp(logits[i][j] / sum_k exp(logits[i][k]))
 *
 * @param logits Logits with shape [batch, classes]
 * @param probs Probabilities with shape [batch, classes]
 *
 * @note The implementation uses maximum-value subtraction before
 *       exponentiation for numerical stability.
 */
inline void softmax(const core::Tensor &logits, core::Tensor &probs) {
  CORE_VERIFY(logits.rank() == 2, "Requires logits with shape [B, C]");
  CORE_VERIFY(probs.rank() == 2, "Requires probs with shape [B,C]");

  [[maybe_unused]] const int batches = logits.shape()[0],
                             classes = logits.shape()[1];

  CORE_VERIFY(batches > 0 && classes > 0,
              "Softmax requires a non-empty batch and class dimension");
  CORE_VERIFY(probs.shape()[0] == batches && probs.shape()[1] == classes,
              "Probabilities must have shape [B, C]");

  CORE_VERIFY(logits.device() == probs.device(),
              "Logits and probabilities must be on the same device");

  CORE_VERIFY(logits.dtype() == probs.dtype(),
              "Logits and probabilities must have the same dtype");
  CORE_VERIFY(logits.dtype() == core::DType::Float32 ||
                  logits.dtype() == core::DType::Float64,
              "Softmax requires floating-point tensors");

  kernels::dispatch(logits.device(), logits.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        kernels::cpu::softmax<T>(MatrixView<const T>(logits),
                                                 MatrixView<T>(probs));
                      else
                        CORE_VERIFY(false, "not implemented yet");
                    });
}
} // namespace ml::ops
