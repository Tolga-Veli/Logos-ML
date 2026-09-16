#pragma once

#include "Core/DType.hpp"
#include "Memory/Device.hpp"

#include "Ops/Views/MatrixView.hpp"
#include "Ops/Views/ScalarView.hpp"
#include "Ops/Views/VectorView.hpp"

#include <utility>

namespace ml::ops {
using backend::MatrixView;
using backend::ScalarView;
using backend::VectorView;

namespace detail {
template <class F> void dispatch_device(memory::Device device, F &&f) {
  using enum memory::DeviceType;

  switch (device.type()) {
  case CPU:
    std::forward<F>(f).template operator()<CPU>();
    return;
  }
}

template <class F> void dispatch_dtype(core::DType dtype, F &&f) {
  using enum core::DType;

  switch (dtype) {
  case Float32:
    std::forward<F>(f).template operator()<float>();
    return;
  case Float64:
    std::forward<F>(f).template operator()<double>();
    return;
  default:
    throw std::logic_error("Dispatching on non-floating point numbers is unsupported");
  }
}
} // namespace detail

template <class F> void dispatch(memory::Device device, core::DType dtype, F &&f) {
  detail::dispatch_device(device, [&]<memory::DeviceType D>() {
    detail::dispatch_dtype(dtype, [&]<class T>() { std::forward<F>(f).template operator()<D, T>(); });
  });
}

} // namespace ml::ops
