#pragma once

#include "Core/Error.hpp"
#include "Core/Tensor.hpp"

#include "Memory/Device.hpp"

#include "Ops/Backend/CPU/XavierUniform.hpp"
#include "Ops/Backend/Dispatch.hpp"

namespace ml::ops::init {

inline void xavier_uniform(core::Tensor &data) {
  if (data.dtype() != core::DType::Float32 && data.dtype() != core::DType::Float64)
    throw DTypeError("Xavier initialization requires a floating-point tensor, got: {}", data.dtype());

  dispatch(data.device(), data.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::xavier_uniform<T>(data);
    else
      UNREACHABLE();
  });
}
} // namespace ml::ops::init
