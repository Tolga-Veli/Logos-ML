#include "Modules/Sequential.hpp"

namespace ml::core {
void Sequential::add(std::unique_ptr<Module> module) {
  this->register_module(*module);
  m_Modules.push_back(std::move(module));
}

void Sequential::forward(const Tensor &X, Tensor &Y) {
  if (m_Modules.empty()) {
    Y = X;
    return;
  }

  Tensor tmp1 = X;
  for (auto &m : m_Modules) {
    Tensor tmp2(m->output_shape(tmp1.shape()), tmp1.dtype());
    m->forward(tmp1, tmp2);
    tmp1 = std::move(tmp2);
  }
  Y = std::move(tmp1);
}

void Sequential::backward(const Tensor &Y, Tensor &X) {
  if (m_Modules.empty()) {
    X = Y;
    return;
  }

  Tensor tmp1 = Y;
  for (auto it = m_Modules.rbegin(); it != m_Modules.rend(); ++it) {
    Tensor tmp2((*it)->input_shape(tmp1.shape()), tmp1.dtype());
    (*it)->backward(tmp1, tmp2);
    tmp1 = std::move(tmp2);
  }

  X = std::move(tmp1);
}

Shape Sequential::output_shape(const Shape &in) const {
  Shape shape = in;
  for (const auto &m : m_Modules)
    shape = m->output_shape(shape);
  return shape;
}

Shape Sequential::input_shape(const Shape &out) const {
  Shape shape = out;
  for (auto it = m_Modules.rbegin(); it != m_Modules.rend(); ++it)
    shape = (*it)->input_shape(shape);
  return shape;
}
} // namespace ml::core
