#pragma once

#include "Core/DType.hpp"
#include "Module.hpp"
#include "Ops/Initializer.hpp"
#include "Parameter.hpp"

#include "Ops/LinearAlgebra.hpp"
#include "Ops/Matmul.hpp"
#include <cmath>

namespace ml::core {

class Linear final : public Module {
public:
  Linear(int in_sz, int out_sz, DType type)
      : m_Weight(Tensor(Shape{in_sz, out_sz}, type)),
        m_Bias(Tensor(Shape{out_sz}, type)) {

    ops::init::xavier_uniform(m_Weight.data);
    m_Bias.data.fill_zero();
  }

  // X = input - [batch, in_sz]
  // W = weights - [in_sz, out_sz]
  // Y = output - [batch, out_sz]
  // b = bias - [out_sz]
  //
  // Y = X * W + b
  void forward(const Tensor &X, Tensor &Y) override {
    m_Input = X;

    const auto batch = X.shape()[0], out_sz = m_Weight.data.shape()[1];
    Shape expected{batch, out_sz};
    if (Y.shape() != expected || Y.dtype() != X.dtype())
      Y = Tensor(expected, X.dtype());

    ops::matmul(ops::Transpose::No, ops::Transpose::No, true, X, m_Weight.data,
                Y);
    ops::add_rowwise_vector(Y, m_Bias.data);
  }

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

  void backward(const Tensor &Y, Tensor &X) override {
    const auto in_sz = m_Weight.data.shape()[0],
               out_sz = m_Weight.data.shape()[1], batch = Y.shape()[0];

    Shape expected{batch, in_sz};
    if (X.shape() != expected || X.dtype() != Y.dtype())
      X = Tensor(expected, Y.dtype());

    ops::matmul(ops::Transpose::No, ops::Transpose::Yes, true, Y, m_Weight.data,
                X);

    if (!m_Weight.has_grad()) {
      m_Weight.grad = Tensor({in_sz, out_sz}, Y.dtype());
      m_Weight.grad->fill_zero();
    }

    ops::matmul(ops::Transpose::Yes, ops::Transpose::No, false, m_Input, Y,
                *m_Weight.grad);

    if (!m_Bias.has_grad()) {
      m_Bias.grad = Tensor({out_sz}, Y.dtype());
      m_Bias.grad->fill_zero();
    }

    ops::sum_rows(Y, *m_Bias.grad);
  }

  std::span<Parameter *const> own_parameters() override { return m_Params; }

  // [batch, out_sz]
  Shape output_shape(const Shape &in) const override {
    return Shape{in[0], m_Weight.data.shape()[1]};
  }

  // [batch, in_sz]
  Shape input_shape(const Shape &out) const override {
    return Shape{out[0], m_Weight.data.shape()[0]};
  }

private:
  Parameter m_Weight, m_Bias;
  std::array<Parameter *, 2> m_Params{&m_Weight, &m_Bias};
  Tensor m_Input;
};
} // namespace ml::core
