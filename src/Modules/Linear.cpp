#include "Modules/Linear.hpp"

namespace ml::core {

Linear::Linear(std::size_t in_sz, std::size_t out_sz, DType type)
    : m_Weight(Tensor(Shape{in_sz, out_sz}, type)), m_Bias(Tensor(Shape{out_sz}, type)) {
  ops::init::xavier_uniform(m_Weight.data);
  ops::fill_zeroes(m_Bias.data);
}

void Linear::forward(const Tensor &X, Tensor &Y) {
  m_Input = X;

  const auto batch = X.shape()[0], out_sz = m_Weight.data.shape()[1];
  Shape expected{batch, out_sz};
  if (Y.shape() != expected || Y.dtype() != X.dtype())
    Y = Tensor(expected, X.dtype());

  ops::matmul(X, ops::Transpose::No, m_Weight.data, ops::Transpose::No, true, Y);
  ops::add_rowwise_vector(Y, m_Bias.data);
}

void Linear::backward(const Tensor &Y, Tensor &X) {
  const auto in_sz = m_Weight.data.shape()[0], out_sz = m_Weight.data.shape()[1], batch = Y.shape()[0];

  Shape expected{batch, in_sz};
  if (X.shape() != expected || X.dtype() != Y.dtype())
    X = Tensor(expected, Y.dtype());

  ops::matmul(Y, ops::Transpose::No, m_Weight.data, ops::Transpose::Yes, true, X);

  if (!m_Weight.has_grad()) {
    m_Weight.grad = Tensor({in_sz, out_sz}, Y.dtype());
    ops::fill_zeroes(m_Weight.grad.value());
  }

  ops::matmul(m_Input, ops::Transpose::Yes, Y, ops::Transpose::No, false, *m_Weight.grad);

  if (!m_Bias.has_grad()) {
    m_Bias.grad = Tensor({out_sz}, Y.dtype());
    ops::fill_zeroes(m_Bias.grad.value());
  }

  ops::sum_rows(Y, *m_Bias.grad);
}
} // namespace ml::core
