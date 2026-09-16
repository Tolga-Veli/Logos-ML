#pragma once

#include "Core/Tensor.hpp"
#include "Ops/Backend/Dispatch.hpp"

namespace ml::ops {
using core::Tensor;

enum class Transpose { No = 0, Yes };

namespace detail {
template <class T> inline void fill(core::Tensor &tensor, T val) {
  T *data = tensor.data<T>();
  if (!data)
    throw std::invalid_argument("fill: Tensor storage was a nullptr");

  if (!tensor.is_contiguous())
    throw std::logic_error("fill: Tensor storage wasn't contiguous");

  std::fill_n(data, tensor.num_elements(), val);
}
} // namespace detail

inline void fill_zeroes(core::Tensor &tensor) {
  detail::dispatch_dtype(tensor.dtype(), [&]<class T>() { detail::fill(tensor, T{0}); });
}

inline void fill_ones(core::Tensor &tensor) {
  detail::dispatch_dtype(tensor.dtype(), [&]<class T>() { detail::fill(tensor, T{1}); });
}

} // namespace ml::ops
