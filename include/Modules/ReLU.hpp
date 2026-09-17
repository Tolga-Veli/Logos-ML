#pragma once

#include "Module.hpp"
#include "Ops/ReLU.hpp"

namespace ml::core {
class ReLU final : public Module {
public:
  // input-output same shape
  Tensor forward(const Tensor &X) override {
    m_Input = X;
    Tensor Y(X.shape());
    ops::relu(X, Y);
    return Y;
  }

  // input-output same shape
  Tensor backward(const Tensor &Y) override {
    Tensor X(Y.shape());
    ops::relu_backward(Y, m_Input, X);
    return X;
  }

private:
  Tensor m_Input;
};
} // namespace ml::core
