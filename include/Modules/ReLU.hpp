#pragma once

#include "Module.hpp"
#include "Ops/ReLU.hpp"

namespace ml::core {
class ReLU final : public Module {
public:
  // input-output same shape
  void forward(const Tensor &X, Tensor &Y) override {
    m_Input = X;
    ops::relu(X, Y);
  }

  // input-output same shape
  void backward(const Tensor &Y, Tensor &X) override {
    ops::relu_backward(Y, m_Input, X);
  }

private:
  Tensor m_Input;
};
} // namespace ml::core
