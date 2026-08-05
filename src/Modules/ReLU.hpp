#pragma once

#include "Math/relu.hpp"
#include "Module.hpp"

namespace ml::core {
template <class T> class ReLU final : public Module<T> {
public:
  Tensor<T> forward(const Tensor<T> &input) override {
    m_Input = input;
    return ops::relu(input);
  }

  Tensor<T> backward(const Tensor<T> &grad_out) override {
    return ops::relu_backward(grad_out, m_Input);
  }

private:
  Tensor<T> m_Input;
};
} // namespace ml::core
