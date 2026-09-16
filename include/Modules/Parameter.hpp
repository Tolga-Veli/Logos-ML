#pragma once

#include <optional>
#include <utility>

#include "Core/Tensor.hpp"
#include "Ops/Utils.hpp"

namespace ml::core {

class Parameter {
public:
  Tensor data;
  std::optional<Tensor> grad;

  explicit Parameter(Tensor _data) : data(std::move(_data)) {}

  [[nodiscard]] bool has_grad() const noexcept { return grad.has_value(); }

  void zero_grad() {
    if (grad.has_value())
      ops::fill_zeroes(grad.value());
  }
};

} // namespace ml::core
