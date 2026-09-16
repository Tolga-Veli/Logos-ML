#pragma once

#include "Modules/Parameter.hpp"

#include <vector>

namespace ml::optim {
template <class T> class SGD {
public:
  SGD(std::vector<core::Parameter *> params, T learning_rate, T momentum = T{0}, T weight_decay = T{0});

  void step();
  void zero_grad();

private:
  std::vector<core::Parameter *> m_Params;
  std::vector<ml::core::Tensor> m_Velocity;
  T m_LearningRate, m_Momentum, m_WeightDecay;
};
} // namespace ml::optim
