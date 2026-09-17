#include "Modules/Sequential.hpp"

namespace ml::core {
void Sequential::add(std::unique_ptr<Module> module) {
  this->register_module(*module);
  m_Modules.push_back(std::move(module));
}

Tensor Sequential::forward(const Tensor &X) {
  Tensor output = X;
  for (const auto &module : m_Modules)
    output = module->forward(output);
  return output;
}

Tensor Sequential::backward(const Tensor &Y) {
  Tensor grad_input = Y;
  for (auto it = m_Modules.rbegin(); it != m_Modules.rend(); ++it)
    grad_input = (*it)->backward(grad_input);
  return grad_input;
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
