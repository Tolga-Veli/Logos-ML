#pragma once

#include "Core/Tensor.hpp"
#include "Ops/Backend/Dispatch.hpp"

namespace ml::ops {
using core::Tensor;

enum class Transpose { No = 0, Yes };

namespace detail {
template <class T> inline void fill(core::Tensor &tensor, T val) {
  T *data = tensor.data<T>();
  CORE_ASSERT(data, "Tensor storage was a nullptr");
  CORE_ASSERT(tensor.is_contiguous(), "Tensor storage wasn't contiguous");

  if (tensor.is_contiguous()) {
    std::fill_n(data, tensor.num_elements(), val);
    return;
  }
}
} // namespace detail

inline void fill_zeroes(core::Tensor &tensor) {
  CORE_ASSERT(tensor.device().type() == memory::DeviceType::CPU, "fill_zeroes requires CPU storage");

  detail::dispatch_dtype(tensor.dtype(), [&]<class T>() { detail::fill(tensor, T{0}); });
}

inline void fill_ones(core::Tensor &tensor) {
  CORE_ASSERT(tensor.device().type() == memory::DeviceType::CPU, "fill_zeroes requires CPU storage");

  detail::dispatch_dtype(tensor.dtype(), [&]<class T>() { detail::fill(tensor, T{1}); });
}
} // namespace ml::ops
