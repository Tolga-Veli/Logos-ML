#pragma once

#include "Core/DType.hpp"
#include "Memory/Device.hpp"

namespace ml::kernels {
namespace detail {

template <class F>
void dispatch_device(memory::Device device, F &&f) {
  using enum memory::DeviceType;
  switch (device.type) {
  case CPU:
    return f.template operator()<CPU>();
  case CUDA:
    f.template operator()<CUDA>();
    return;
  }

  CORE_VERIFY(false, "Unsupported device type");
}

template <class F> void dispatch_dtype(core::DType dtype, F &&f) {
  using enum core::DType;
  switch (dtype) {
  case Float32:
    f.template operator()<float>();
    return;
  case Float64:
    f.template operator()<double>();
    return;
  case Int32:
    CORE_VERIFY(false, "This kernel does not support Int32 tensors");
  }

  CORE_VERIFY(false, "Unsupported dtype");
}
} // namespace detail

template <class F>
void dispatch(memory::Device device, core::DType dtype, F &&f) {
  detail::dispatch_device(device, [&]<memory::DeviceType D>() {
    detail::dispatch_dtype(dtype,
                           [&]<class T>() { f.template operator()<D, T>(); });
  });
}

} // namespace ml::kernels
