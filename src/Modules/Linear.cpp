#include "Modules/Linear.hpp"

#include "Ops/Initializers/XavierUniform.hpp"
#include "Ops/LinearAlgebra.hpp"
#include "Ops/Matmul.hpp"

namespace ml::core {

Linear::Linear(std::size_t in_sz, std::size_t out_sz, DType type)
    : m_Weight(Tensor(Shape{in_sz, out_sz}, type)), m_Bias(Tensor(Shape{out_sz}, type)) {
  ops::init::xavier_uniform(m_Weight.value());
  ops::fill_zeroes(m_Bias.value());
}

Tensor Linear::forward(const Tensor &X) {
  m_Input = X;
  Tensor Y = ops::matmul(X, ops::Transpose::No, m_Weight.value(), ops::Transpose::No);
  ops::add_rowwise_vector(Y, m_Bias.value());
  return Y;
}

Tensor Linear::backward(const Tensor &Y) {
  const auto in_sz = m_Weight.value().shape()[0], out_sz = m_Weight.value().shape()[1];
  Tensor X = ops::matmul(Y, ops::Transpose::No, m_Weight.value(), ops::Transpose::Yes);

  if (!m_Weight.has_grad()) {
    m_Weight.initialize_grad(Tensor{{in_sz, out_sz}, Y.dtype()});
    ops::fill_zeroes(m_Weight.grad());
  }

  ops::matmul(m_Input, ops::Transpose::Yes, Y, ops::Transpose::No, false, m_Weight.grad());

  if (!m_Bias.has_grad()) {
    m_Bias.initialize_grad(Tensor{{out_sz}, Y.dtype()});
    ops::fill_zeroes(m_Bias.grad());
  }

  ops::sum_rows(Y, m_Bias.grad());
  return X;
}
} // namespace ml::core
