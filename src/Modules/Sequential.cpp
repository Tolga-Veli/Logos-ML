#include "Modules/Sequential.hpp"

namespace ml::core {
void Sequential::add(std::unique_ptr<Module> module) {
  register_module(std::to_string(children().size()), std::move(module));
}

Tensor Sequential::forward(const Tensor &X) {
  Tensor output = X;
  for (const auto &child : children())
    output = child.module->forward(output);
  return output;
}

Tensor Sequential::backward(const Tensor &Y) {
  Tensor grad_input = Y;
  const auto modules = children();
  for (auto it = modules.rbegin(); it != modules.rend(); ++it)
    grad_input = it->module->backward(grad_input);
  return grad_input;
}

Shape Sequential::output_shape(const Shape &in) const {
  Shape shape = in;
  for (const auto &child : children())
    shape = child.module->output_shape(shape);
  return shape;
}

Shape Sequential::input_shape(const Shape &out) const {
  Shape shape = out;
  const auto modules = children();
  for (auto it = modules.rbegin(); it != modules.rend(); ++it)
    shape = it->module->input_shape(shape);
  return shape;
}
} // namespace ml::core
