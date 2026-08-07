#pragma once

#include "Core/Tensor.hpp"
#include "Kernels/CPU/Softmax.hpp"
#include "Memory/Device.hpp"
#include "Ops/Kernels/Dispatch.hpp"

#include <cmath>

namespace ml::ops {
using core::Tensor;
// input - [batch, classes]
// output - [batch, classes]
inline void softmax(const Tensor &input, Tensor &output) {
  CORE_VERIFY(input.device() == output.device(),
              "Input vectors do not live on the same device");
  CORE_VERIFY(input.dtype() == output.dtype(),
              "Input vectors do not share the same dtype");

  kernels::dispatch(input.device(), input.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        kernels::cpu::softmax<T>(input, output);
                      else
                        CORE_VERIFY(false, "not implemented yet");
                    });
}
} // namespace ml::ops
