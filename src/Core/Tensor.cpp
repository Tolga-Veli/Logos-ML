#include "Core/Tensor.hpp"

#include "Ops/Matmul.hpp"

namespace ml::core {
Tensor Tensor::operator+(const Tensor &other) {
  const auto &s1 = shape(), &s2 = other.shape();
  auto shape = ops::detail::broadcast_shapes(s1, s2);

  Tensor out(shape);
  // TODO:
  return out;
}

Tensor Tensor::operator*(const Tensor &other) {
  return ops::matmul(*this, ops::Transpose::No, other, ops::Transpose::No);
}

} // namespace ml::core
