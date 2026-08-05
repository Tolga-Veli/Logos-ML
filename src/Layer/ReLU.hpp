#pragma once

#include "Math/ops.hpp"
#include "Module.hpp"

namespace ml::core {
template <class T> class ReLU final : public Module<T> {
public:
  using Tensor = Tensor<T>;

  Tensor forward(const Tensor &input) override {
    m_Input = input;
    return ops::relu(input);
  }

  Tensor backward(const Tensor &grad_out) override {
    return ops::relu_backward(grad_out, m_Input);
  }

  std::span<Parameter<T> *const> parameters() const noexcept override {
    return {};
  }
  void zero_grad() const noexcept override {};

private:
  Tensor m_Input;
};
} // namespace ml::core
