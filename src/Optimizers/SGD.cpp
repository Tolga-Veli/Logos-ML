#include "Optimizers/SGD.hpp"
#include "Ops/Utils.hpp"

namespace ml::optim {
template <class T> SGD<T>::SGD(std::vector<core::Parameter *> params, T learning_rate, T momentum, T weight_decay)
    : m_Params(std::move(params)), m_LearningRate(learning_rate), m_Momentum(momentum), m_WeightDecay(weight_decay) {

  CORE_VERIFY(m_LearningRate >= T{0}, "learning_rate must be non-negative");
  CORE_VERIFY(m_Momentum >= T{0} && m_Momentum < T{1}, "momentum must be in [0, 1)");
  CORE_VERIFY(m_WeightDecay >= T{0}, "weight_decay must be non-negative");

  for (const auto *p : m_Params) {
    CORE_VERIFY(p != nullptr, "optimizer parameters cannot be null");
    CORE_VERIFY(p->data.dtype() == core::dtype_of_v<T>, "optimizer type must match parameter dtype");
  }

  if (m_Momentum > T{0}) {
    m_Velocity.reserve(m_Params.size());
    for (auto *p : m_Params) {
      m_Velocity.emplace_back(p->data.shape(), p->data.dtype());
      ops::fill_zeroes(m_Velocity.back());
    }
  }
}

template <class T> void SGD<T>::step() {
  for (std::size_t i = 0; i < m_Params.size(); i++) {
    auto *p = m_Params[i];
    if (!p->has_grad())
      continue;

    CORE_VERIFY(p->grad->shape() == p->data.shape(), "gradient shape must match parameter shape");
    CORE_VERIFY(p->grad->dtype() == p->data.dtype(), "gradient dtype must match parameter dtype");

    T *w = p->data.data<T>();
    const T *grad = p->grad->data<T>();
    const int n = p->data.num_elements();

    for (int j = 0; j < n; j++) {
      // Coupled L2 regularization: differentiating (lambda/2)||w||^2 adds
      // lambda*w to the data gradient.
      const T g = grad[j] + m_WeightDecay * w[j];

      if (m_Momentum > T{0}) {
        // Classical momentum accumulates an exponentially weighted velocity.
        // Sutskever et al. (2013): proceedings.mlr.press/v28/sutskever13.html
        T *vel = m_Velocity[i].data<T>();
        vel[j] = m_Momentum * vel[j] + g;
        w[j] -= m_LearningRate * vel[j];
      } else
        w[j] -= m_LearningRate * g;
    }
  }
}

template <class T> void SGD<T>::zero_grad() {
  for (auto *p : m_Params)
    p->zero_grad();
}

} // namespace ml::optim

template class ml::optim::SGD<float>;
