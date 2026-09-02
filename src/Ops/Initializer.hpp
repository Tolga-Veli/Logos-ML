#pragma once

#include "Core/Tensor.hpp"
#include "Memory/Device.hpp"
#include "Ops/Kernels/Dispatch.hpp"

#include <cmath>
#include <random>

namespace ml::ops::init {
using core::Tensor;

namespace detail {
template <class T> inline void xavier_uniform_impl(Tensor &data) {
  auto data_shape = data.shape();
  // Glorot uniform chooses Var(W) = 2/(fan_in + fan_out), which for a
  // uniform U[-a,a] gives a = sqrt(6/(fan_in + fan_out)).  The goal is to
  // keep activation and gradient variance from drifting across layers.
  // Glorot & Bengio (2010): proceedings.mlr.press/v9/glorot10a.html
  const T lim = std::sqrt(T{6} / static_cast<T>(data_shape[0] + data_shape[1]));

  std::uniform_real_distribution<T> dist(-lim, lim);
  static std::mt19937 rng{std::random_device{}()};

  auto *w = data.data<T>();
  for (int i = 0; i < data.num_elements(); i++)
    w[i] = dist(rng);
}
} // namespace detail

inline void xavier_uniform(Tensor &data) {
  CORE_VERIFY(data.rank() == 2,
              "Xavier initialization requires [fan_in, fan_out]");
  CORE_VERIFY(data.dtype() == core::DType::Float32 ||
                  data.dtype() == core::DType::Float64,
              "Xavier initialization requires a floating-point tensor");
  CORE_VERIFY(data.shape()[0] > 0 && data.shape()[1] > 0,
              "Xavier initialization requires positive fan sizes");

  kernels::dispatch(data.device(), data.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        detail::xavier_uniform_impl<T>(data);
                      else
                        CORE_VERIFY(false, "unreachable");
                    });
}
} // namespace ml::ops::init
