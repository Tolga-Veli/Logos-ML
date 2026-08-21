#pragma once

#include "Modules/Parameter.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace ml::optim {
template <class T> class SGD {
public:
  SGD(std::vector<core::Parameter *> params, T learning_rate, T momentum = T{0},
      T weight_decay = T{0})
      : m_Params(std::move(params)), m_LearningRate(learning_rate),
        m_Momentum(momentum), m_WeightDecay(weight_decay) {

    if (m_Momentum > T{0}) {
      m_Velocity.reserve(m_Params.size());
      for (auto *p : m_Params) {
        m_Velocity.emplace_back(p->data.shape(), p->data.dtype());
        m_Velocity.back().fill_zero();
      }
    }
  }

  void step() {
    for (int i = 0; i < static_cast<int>(m_Params.size()); i++) {
      auto *p = m_Params[i];
      if (!p->has_grad())
        continue;

      T *w = p->data.data<T>();
      const T *grad = p->grad->data<T>();
      const int n = p->data.num_elements();

      for (int j = 0; j < n; j++) {
        // clipped gradient + optional L2 penalty
        const T g = grad[j] + m_WeightDecay * w[j];

        if (m_Momentum > T{0}) {
          T *vel = m_Velocity[i].data<T>();
          vel[j] = m_Momentum * vel[j] + g;
          w[j] -= m_LearningRate * vel[j];
        } else
          w[j] -= m_LearningRate * g;
      }
    }
  }

  void zero_grad() {
    for (auto *p : m_Params)
      p->zero_grad();
  }

private:
  std::vector<core::Parameter *> m_Params;
  std::vector<ml::core::Tensor> m_Velocity;
  T m_LearningRate, m_Momentum, m_WeightDecay;
};
} // namespace ml::optim
