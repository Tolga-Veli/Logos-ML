#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"
#include "Memory/Device.hpp"
#include "Ops/Kernels/CPU/ReLU.hpp"
#include "Ops/Kernels/Dispatch.hpp"

#include <cmath>

namespace ml::ops {
using core::Tensor;

/*
 *  @brief Computes the ReLU elementwise
 *
 *  Computes:
 *
 *    out[i] = max(input[i], 0)
 *
 *  for every element of the input tensor
 *
 *  @param input Input tensor
 *  @param out Output tensor
 */
inline void relu(const Tensor &x, Tensor &out) {
  CORE_ASSERT(x.shape() == out.shape(),
              "Requires input and output to have the same shape");

  CORE_ASSERT(x.device() == out.device(),
              "Requires input and output to be on the same device");

  CORE_ASSERT(x.dtype() == out.dtype(),
              "Requires input and output to have the same dtype");

  CORE_ASSERT(x.is_contiguous() && out.is_contiguous(),
              "Requires contiguous tensors");

  kernels::dispatch(x.device(), x.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        kernels::cpu::relu<T>(x, out);
                      else
                        CORE_ASSERT(false, "not implemented yet");
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
inline void relu_backward(const Tensor &grad_out, const Tensor &x,
                          Tensor &grad_in) {

  CORE_ASSERT(grad_out.shape() == x.shape() &&
                  grad_out.shape() == grad_in.shape(),
              "Tensors must have the same shape");

  CORE_ASSERT(grad_out.device() == x.device() &&
                  grad_out.device() == grad_in.device(),
              "Tensors must live on the same device");

  CORE_ASSERT(grad_out.dtype() == x.dtype() &&
                  grad_out.dtype() == grad_in.dtype(),
              "Tensors must share the same dtype");

  CORE_ASSERT(grad_out.is_contiguous() && x.is_contiguous() &&
                  grad_in.is_contiguous(),
              "Requires contiguous tensors");

  kernels::dispatch(x.device(), x.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        kernels::cpu::relu_backward<T>(grad_out, x, grad_in);
                      else
                        CORE_ASSERT(false, "not implemented yet");
                    });
}
} // namespace ml::ops
