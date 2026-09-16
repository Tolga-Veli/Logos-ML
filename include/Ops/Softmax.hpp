#pragma once

#include "Core/Tensor.hpp"

#include "Ops/Backend/CPU/Softmax.hpp"
#include "Ops/Backend/Dispatch.hpp"
#include "Ops/Validation.hpp"

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
  require_same_dtype(logits, probs);
  require_same_device(logits, probs);

  dispatch(logits.device(), logits.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::softmax<T>(logits, probs);
    else
      UNREACHABLE();
  });
}

inline core::Tensor softmax(const core::Tensor &logits) {
  core::Tensor probs(logits.shape(), logits.dtype());
  softmax(logits, probs);
  return probs;
}
} // namespace ml::ops
