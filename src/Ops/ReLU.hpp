#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"
#include "Memory/Device.hpp"
#include "Ops/Kernels/CPU/ReLU.hpp"
#include "Ops/Kernels/Dispatch.hpp"

#include <cmath>

namespace ml::ops {
using core::Tensor;

// out[i] = max(x[i], 0)
// x and out must be same shape and x must be contiguous
inline void relu(const Tensor &x, Tensor &out) {
  CORE_VERIFY(x.device() == out.device(), "Invalid device");
  CORE_VERIFY(x.dtype() == out.dtype(), "Invalid dtype");

  kernels::dispatch(x.device(), x.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        kernels::cpu::relu<T>(x, out);
                      else
                        CORE_VERIFY(false, "not implemented yet");
                    });
}

// grad_input[i] = grad_out[i] if x[i] > 0, else 0
// grad_in and grad_out must be same shape
// grad_out and x must be contiguous
inline void relu_backward(const Tensor &grad_out, const Tensor &x,
                          Tensor &grad_in) {
  CORE_VERIFY(grad_out.device() == x.device() &&
                  grad_out.device() == grad_in.device(),
              "Invalid device");
  CORE_VERIFY(grad_out.dtype() == x.dtype() &&
                  grad_out.dtype() == grad_in.dtype(),
              "Invalid dtype");

  kernels::dispatch(x.device(), x.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        kernels::cpu::relu_backward<T>(grad_out, x, grad_in);
                      else
                        CORE_VERIFY(false, "not implemented yet");
                    });
}
} // namespace ml::ops
