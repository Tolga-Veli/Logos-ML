#pragma once

#include "Module.hpp"
#include "Parameter.hpp"

#include "Math/matrix-matrix.hpp"
#include "Math/matrix-vector.hpp"

namespace ml::core {

template <class T> class Linear final : public Module<T> {
public:
  using Tensor = Tensor<T>;

  Linear(std::size_t input, std::size_t out);

  // output = input * weight^T + bias
  Tensor forward(const Tensor &input) override {
    const auto batch = input.shape()[0];
    const auto out_sz = m_Weight.data.shape()[0];

    m_Input = input;

    Tensor output({batch, out_sz});

    linalg::matmul(linalg::Transpose::No, linalg::Transpose::Yes, input,
                   m_Weight.data, output);

    linalg::add_row_vector(output, m_Bias.data);

    return output;
  }

  Tensor backward(const Tensor &grad_out) override {
    const auto batch = grad_out.shape()[0];
    const auto in_sz = m_Weight.data.shape()[1];
    const auto out_sz = m_Weight.data.shape()[0];

    Tensor grad_in({batch, in_sz});

    linalg::matmul(linalg::Transpose::No, linalg::Transpose::No, grad_out,
                   m_Weight.data, grad_in);

    if (!m_Weight.has_grad()) {
      m_Weight.grad = Tensor({out_sz, in_sz});
      m_Weight.grad->fill(T{0});
    }

    linalg::matmul(linalg::Transpose::Yes, linalg::Transpose::No, T{1},
                   grad_out, m_Input, T{1}, *m_Weight.grad);

    if (!m_Bias.has_grad()) {
      m_Bias.grad = Tensor({out_sz});
      m_Bias.grad->fill(T{0});
    }

    linalg::sum_rows(grad_out, *m_Bias.grad);

    return grad_in;
  }

  std::span<Parameter<T> *const> parameters() override { return m_Params; }

  void zero_grad() noexcept override {
    m_Weight.zero_grad();
    m_Bias.zero_grad();
  }

private:
  Parameter<T> m_Weight, m_Bias;
  std::array<Parameter<T> *, 2> m_Params{&m_Weight, &m_Bias};
  Tensor m_Input;
};
} // namespace ml::core
