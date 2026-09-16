#pragma once

#include "Core/Error.hpp"
#include "Core/Tensor.hpp"

#include <cstddef>
#include <type_traits>

namespace ml::ops {

template <class... Tensors>
  requires((std::same_as<std::remove_cvref_t<Tensors>, core::Tensor>) && ...)
inline void require_same_dtype(const core::Tensor &first, const Tensors &...rest) {
  auto check = [&](const core::Tensor &tensor) {
    if (tensor.dtype() != first.dtype())
      throw DTypeError("Tensors must have the same dtype; expected {}, got {}", first.dtype(), tensor.dtype());
  };

  (check(rest), ...);
}

template <class... Tensors>
  requires((std::same_as<std::remove_cvref_t<Tensors>, core::Tensor>) && ...)
inline void require_same_device(const core::Tensor &first, const Tensors &...rest) {
  auto check = [&](const core::Tensor &tensor) {
    if (tensor.device() != first.device())
      throw DeviceError("Tensors must be on the same device; expected {}, got {}", first.device(), tensor.device());
  };

  (check(rest), ...);
}

inline void require_rank(const core::Tensor &tensor, std::size_t rank) {
  if (tensor.rank() != rank)
    throw ShapeError("Expected rank {}, got {}", rank, tensor.rank());
}

inline void require_num_elements(const core::Tensor &tensor, std::size_t count) {
  if (tensor.num_elements() != count)
    throw ShapeError("Expected {} elements, got {}", count, tensor.num_elements());
}
} // namespace ml::ops
