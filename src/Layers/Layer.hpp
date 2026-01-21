#pragma once

#include "LinearAlgebra/Matrix.hpp"

namespace Logos::NeuralNet {
template <class T> class ILayer {
public:
  ILayer() = default;
  virtual ~ILayer() = default;

  virtual void Forward(const linalg::Matrix<T> &in, linalg::Matrix<T> &out) = 0;
  virtual void Backward(const linalg::Matrix<T> &in,
                        linalg::Matrix<T> &out) = 0;

  virtual void ZeroGrads() = 0;
  virtual void GradientDescentStep(float learning_rate) = 0;
};

} // namespace Logos::NeuralNet
