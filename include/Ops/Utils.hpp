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

inline core::Tensor broadcast_to(const core::Tensor &tensor, const core::Shape &new_shape) {
  const auto &shape = tensor.shape();
  const auto &strides = tensor.strides();

  if (new_shape.rank() < shape.rank())
    throw ShapeError("broadcast_to: cannot broadcast rank {} tensor to rank {}", shape.rank(), new_shape.rank());

  core::Strides new_strides(new_shape.rank());

  const std::size_t offset = new_shape.rank() - shape.rank();
  for (std::size_t i = 0; i < new_shape.rank(); i++) {
    if (i < offset) {
      new_strides[i] = 0;
      continue;
    }

    const std::size_t prev = i - offset;

    if (shape[prev] == new_shape[i])
      new_strides[i] = strides[prev];
    else if (shape[prev] == 1)
      new_strides[i] = 0;
    else
      throw ShapeError("broadcast_to: cannot broadcast dimension {} from {} to {}", prev, shape[prev], new_shape[i]);
  }

  return tensor.as_strided(new_shape, new_strides);
}

} // namespace ml::ops
