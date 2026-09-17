#pragma once

#include "Core/DType.hpp"
#include "Modules/Module.hpp"
#include "Modules/Parameter.hpp"

#include <array>

namespace ml::core {

class Linear final : public Module {
public:
  Linear(std::size_t in_sz, std::size_t out_sz, DType type = DType::Float32);

  // X = input - [batch, in_sz]
  // W = weights - [in_sz, out_sz]
  // Y = output - [batch, out_sz]
  // b = bias - [out_sz]
  //
  // Y = X * W + b
  void forward(const Tensor &X, Tensor &Y) override;

  // X = input - [batch, in_sz]
  // W = weights - [in_sz, out_sz]
  // Y = output - [batch, out_sz]
  // b = bias - [out_sz]
  //
  // G = dL/dY - upstream gradient - [batch, out_sz]
  //
  // dL/dX = dL/dY * dY/dX = G * W^T
  // dL/dX - [batch, in_sz]
  //
  // we use W^T since G - [batch, out_sz] and W^T - [out_sz, in_sz]
  //
  // dL/dW = dL/dY * dY/dW = X^T * G
  // dL/dW - [in_sz, out_sz]
  //
  // we use X^T since X^T - [in_sz, batch] and G - [batch, out_sz]
  //
  // dL/db = dL/dY * dY/db = sum_{batch} dL/dY = sum_{batch} G
  //
  // the partial derivative of the loss w.r.t. the bias is just the sum of the
  // partial derivatives of the loss with respect to the output Y

  void backward(const Tensor &Y, Tensor &X) override;

  [[nodiscard]] std::span<Parameter *const> own_parameters() override { return m_Params; }

  [[nodiscard]] Shape input_shape(const Shape &out) const override {
    return Shape{out[0], m_Weight.value().shape()[0]};
  }
  [[nodiscard]] Shape output_shape(const Shape &in) const override { return Shape{in[0], m_Weight.value().shape()[1]}; }

private:
  Parameter m_Weight, m_Bias;
  std::array<Parameter *, 2> m_Params{&m_Weight, &m_Bias};
  Tensor m_Input;
};
} // namespace ml::core
