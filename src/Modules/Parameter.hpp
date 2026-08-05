#pragma once

#include "Core/Tensor.hpp"

namespace ml::core {

template <class T> class Parameter {
public:
  Tensor<T> data;
  std::optional<Tensor<T>> grad;

  explicit Parameter(Tensor<T> _data) : data(std::move(_data)) {}

  bool has_grad() const noexcept { return grad.has_value(); }

  void zero_grad() {
    if (grad.has_value())
      grad->fill(T{0});
  }
};

} // namespace ml::core
