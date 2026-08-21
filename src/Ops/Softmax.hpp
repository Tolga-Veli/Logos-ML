#pragma once

#include "Core/Tensor.hpp"
#include "Kernels/CPU/Softmax.hpp"
#include "Memory/Device.hpp"
#include "Ops/Kernels/Dispatch.hpp"

#include <cmath>

namespace ml::ops {
using core::Tensor;
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
inline void softmax(const Tensor &logits, Tensor &probs) {
  CORE_ASSERT(logits.rank() == 2, "Requires logits with shape [B, C]");
  CORE_ASSERT(probs.rank() == 2, "Requires probs with shape [B,C]");

  [[maybe_unused]] const int batches = logits.shape()[0],
                             classes = logits.shape()[1];

  CORE_ASSERT(probs.shape()[0] == batches && probs.shape()[1] == classes,
              "Probabilities must have shape [B, C]");

  CORE_ASSERT(logits.device() == probs.device(),
              "Logits and probabilities must be on the same device");

  CORE_ASSERT(logits.dtype() == probs.dtype(),
              "Logits and probabilities must have the same dtype");

  kernels::dispatch(logits.device(), logits.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        kernels::cpu::softmax<T>(MatrixView<const T>(logits),
                                                 MatrixView<T>(probs));
                      else
                        CORE_ASSERT(false, "not implemented yet");
                    });
}
} // namespace ml::ops
