#pragma once

#include "Module.hpp"

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace ml::core {
class Sequential final : public Module {
public:
  Sequential() = default;

  void add(std::unique_ptr<Module> module) {
    this->register_module(*module);
    m_Modules.push_back(std::move(module));
  }

  template <class ModuleType, class... Args>
    requires std::is_base_of_v<Module, ModuleType>
  void add(Args &&...args) {
    add(std::make_unique<ModuleType>(std::forward<Args>(args)...));
  }

  void forward(const Tensor &X, Tensor &Y) override {
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

  void backward(const Tensor &Y, Tensor &X) override {
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

  Shape output_shape(const Shape &in) const override {
    Shape shape = in;
    for (const auto &m : m_Modules)
      shape = m->output_shape(shape);
    return shape;
  }

  Shape input_shape(const Shape &out) const override {
    Shape shape = out;
    for (auto it = m_Modules.rbegin(); it != m_Modules.rend(); ++it)
      shape = (*it)->input_shape(shape);
    return shape;
  }

private:
  std::vector<std::unique_ptr<Module>> m_Modules;
};
} // namespace ml::core
