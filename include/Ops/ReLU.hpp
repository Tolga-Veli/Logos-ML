#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"
#include "Memory/Device.hpp"
#include "Ops/Backend/CPU/ReLU.hpp"
#include "Ops/Backend/Dispatch.hpp"

namespace ml::ops {
/*
 *  @brief Computes out[i] = max(input[i], 0)
 *
 *  for every element of the input tensor
 *
 *  @param input Input tensor
 *  @param out Output tensor
 */
inline void relu(const core::Tensor &x, core::Tensor &out) {

  CORE_VERIFY(x.dtype() == out.dtype(), "Requires input and output to have the same dtype");

  dispatch(x.device(), x.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::relu<T>(x, out);
    else
      UNREACHABLE();
  });
}

/*
 * @brief Computes the gradient of ReLU with respect to its input
 *
 * Computes:
 *
 *   grad_in[i] = grad_out[i]    if x[i] > 0
 *                0              otherwise
 *
 * @param grad_out Gradient of the output tensor.
 * @param x Input tensor used during the forward pass.
 * @param grad_in Output gradient with respect to x.
 */
inline void relu_backward(const core::Tensor &grad_out, const core::Tensor &x, core::Tensor &grad_in) {

  CORE_VERIFY(grad_out.dtype() == x.dtype() && grad_out.dtype() == grad_in.dtype(),
              "Tensors must share the same dtype");

  dispatch(x.device(), x.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::relu_backward<T>(grad_out, x, grad_in);
    else
      UNREACHABLE();
  });
}
} // namespace ml::ops
