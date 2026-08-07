#pragma once

#include "Core/DType.hpp"
#include "Memory/Device.hpp"
#include <utility>

namespace ml::kernels {
namespace detail {

template <class F>
decltype(auto) dispatch_device(memory::Device device, F &&f) {
  using enum memory::DeviceType;
  switch (device.type) {
  case CPU:
    return f.template operator()<CPU>();
  case CUDA:
    return f.template operator()<CUDA>();
  }

  std::unreachable();
}

template <class F> decltype(auto) dispatch_dtype(core::DType dtype, F &&f) {
  using enum core::DType;
  switch (dtype) {
  case Float32:
    return f.template operator()<float>();
  case Float64:
    return f.template operator()<double>();
  default:
    std::unreachable();
    //    return f.template operator()<int32_t>();
  }
}
} // namespace detail

template <class F>
decltype(auto) dispatch(memory::Device device, core::DType dtype, F &&f) {
  return detail::dispatch_device(device, [&]<memory::DeviceType D>() {
    detail::dispatch_dtype(dtype,
                           [&]<class T>() { f.template operator()<D, T>(); });
  });
}

} // namespace ml::kernels
