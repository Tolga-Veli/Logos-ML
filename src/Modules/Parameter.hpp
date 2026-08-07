#pragma once

#include "Core/Tensor.hpp"

namespace ml::core {

class Parameter {
public:
  Tensor data;
  std::optional<Tensor> grad;

  explicit Parameter(Tensor _data) : data(std::move(_data)) {}

  bool has_grad() const noexcept { return grad.has_value(); }

  void zero_grad() {
    if (grad.has_value())
      grad->fill_zero();
  }
};

} // namespace ml::core
