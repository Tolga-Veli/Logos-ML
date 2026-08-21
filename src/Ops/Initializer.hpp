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
  const T lim = std::sqrt(T{6} / static_cast<T>(data_shape[0] + data_shape[1]));

  std::uniform_real_distribution<T> dist(-lim, lim);
  static std::mt19937 rng{std::random_device{}()};

  auto *w = data.data<T>();
  for (int i = 0; i < data.num_elements(); i++)
    w[i] = dist(rng);
}
} // namespace detail

inline void xavier_uniform(Tensor &data) {
  kernels::dispatch(data.device(), data.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        detail::xavier_uniform_impl<T>(data);
                      else
                        CORE_ASSERT(false, "unreachable");
                    });
}
} // namespace ml::ops::init
