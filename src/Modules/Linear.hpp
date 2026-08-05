#pragma once

#include "Module.hpp"
#include "Parameter.hpp"

#include "Math/matrix-matrix.hpp"
#include "Math/matrix-vector.hpp"
#include <random>

namespace ml::core {

template <class T> class Linear final : public Module<T> {
public:
  Linear(int in_sz, int out_sz)
      : m_Weight(Tensor<T>({in_sz, out_sz})), m_Bias(Tensor<T>({out_sz})) {

    const T lim = std::sqrt(6.0f / static_cast<T>(in_sz + out_sz));

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-lim, lim);

    auto *w = m_Weight.data.data();
    for (int i = 0; i < m_Weight.data.num_elements(); i++)
      w[i] = dist(rng);

    m_Bias.data.fill(T{0});
  }

  // X = input - [batch, in_sz]
  // W = weights - [in_sz, out_sz]
  // Y = output - [batch, out_sz]
  // b = bias - [out_sz]
  //
  // Y = X * W + b
  Tensor<T> forward(const Tensor<T> &input) override {
    const auto batch = input.shape()[0];
    const auto out_sz = m_Weight.data.shape()[1];

    m_Input = input;

    Tensor output({batch, out_sz});

    linalg::matmul(linalg::Transpose::No, linalg::Transpose::No, input,
                   m_Weight.data, output);

    linalg::add_row_vector(output, m_Bias.data);

    return output;
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

  Tensor<T> backward(const Tensor<T> &grad_out) override {
    const auto batch = grad_out.shape()[0];
    const auto in_sz = m_Weight.data.shape()[0];
    const auto out_sz = m_Weight.data.shape()[1];

    Tensor grad_in({batch, in_sz});

    linalg::matmul(linalg::Transpose::No, linalg::Transpose::Yes, grad_out,
                   m_Weight.data, grad_in);

    if (!m_Weight.has_grad()) {
      m_Weight.grad = Tensor({in_sz, out_sz});
      m_Weight.grad->fill(T{0});
    }

    linalg::matmul(linalg::Transpose::Yes, linalg::Transpose::No, T{1}, m_Input,
                   grad_out, T{1}, *m_Weight.grad);

    if (!m_Bias.has_grad()) {
      m_Bias.grad = Tensor({out_sz});
      m_Bias.grad->fill(T{0});
    }

    linalg::sum_rows(grad_out, *m_Bias.grad);

    return grad_in;
  }

  std::span<Parameter<T> *const> own_parameters() override { return m_Params; }

private:
  Parameter<T> m_Weight, m_Bias;
  std::array<Parameter<T> *, 2> m_Params{&m_Weight, &m_Bias};
  Tensor<T> m_Input;
};
} // namespace ml::core
