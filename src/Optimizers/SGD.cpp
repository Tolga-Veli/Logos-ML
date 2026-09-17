#include "Optimizers/SGD.hpp"
#include "Core/Error.hpp"
#include "Ops/Utils.hpp"

namespace ml::optim {
template <class T> SGD<T>::SGD(std::vector<core::Parameter *> params, T learning_rate, T momentum, T weight_decay)
    : m_Params(std::move(params)), m_LearningRate(learning_rate), m_Momentum(momentum), m_WeightDecay(weight_decay) {

  if (m_LearningRate <= T{0})
    throw std::invalid_argument("SGD: Learning rate must be positive");

  if (m_Momentum < T{0} || m_Momentum >= T{1})
    throw std::invalid_argument("SGD: Moment must be in [0,1)");

  if (m_WeightDecay < T{0})
    throw std::invalid_argument("SGD: Weight decay must be non-negative");

  for (const auto *p : m_Params) {
    if (!p)
      throw std::invalid_argument("SGD: optimizer parameters cannot be null");

    if (p->value().dtype() != core::dtype_of_v<T>)
      throw DTypeError("SGD: optimizer type must match parameter dtype");
  }

  if (m_Momentum > T{0}) {
    m_Velocity.reserve(m_Params.size());

    for (auto *p : m_Params) {
      m_Velocity.emplace_back(p->value().shape(), p->value().dtype());
      ops::fill_zeroes(m_Velocity.back());
    }
  }
}

template <class T> void SGD<T>::step() {
  for (std::size_t i = 0; i < m_Params.size(); i++) {
    auto *p = m_Params[i];
    if (!p->has_grad())
      continue;

    if (p->grad().shape() != p->value().shape())
      throw ShapeError("SGD: gradient shape must match parameter shape");
    if (p->grad().dtype() != p->value().dtype())
      throw DTypeError("SGD: gradient dtype must match parameter dtype");

    T *w = p->value().data<T>();
    const T *grad = p->grad().data<T>();
    const auto n = p->value().num_elements();

    for (std::size_t j = 0; j < n; j++) {
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

template <class T> void SGD<T>::zero_grad() {
  for (auto *p : m_Params)
    p->zero_grad();
}
} // namespace ml::optim

template class ml::optim::SGD<float>;
