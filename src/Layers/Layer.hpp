#pragma once

#include "Math/MatrixView.hpp"

namespace Logos::NeuralNet {
template <class T> class ILayer {
public:
  ILayer() = default;
  virtual ~ILayer() = default;

  virtual void Forward(linalg::MatrixView<const T> in,
                       linalg::MatrixView<T> out, bool cache) = 0;
  virtual void Backward(linalg::MatrixView<const T> prev,
                        linalg::MatrixView<T> curr) = 0;

  virtual void ZeroGrads() = 0;
  virtual void GradientDescentStep(float learning_rate) = 0;
};

} // namespace Logos::NeuralNet
