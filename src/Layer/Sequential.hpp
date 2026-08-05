#pragma once

#include "Module.hpp"

#include <memory>
#include <type_traits>
#include <vector>

namespace ml::core {
template <class T> class Sequential final : Module<T> {
public:
  using Tensor = Tensor<T>;

  Sequential() = default;

  template <class ModuleType, class... Args> void add(Args &&...args) {
    static_assert(std::is_base_of_v<Module<T>, ModuleType>,
                  "ModuleType must derive from Module<T>");

    m_Modules.push_back(
        std::make_unique<ModuleType>(std::forward<Args>(args)...));
  }

  Tensor forward(const Tensor &input) override {
    Tensor x = input;
    for (auto &m : m_Modules)
      x = m->forward(x);
    return x;
  }

  Tensor backward(const Tensor &grad_out) override {
    Tensor grad = grad_out;
    for (auto it = m_Modules.rbegin(); it != m_Modules.rend(); it++)
      grad = (*it)->backward(grad);
    return grad;
  }

  std::span<Parameter<T> *const> parameters() override {
    m_Params.clear();
    for (auto &m : m_Modules)
      for (auto *p : m->parameters())
        m_Params.push_back(p);

    return m_Params;
  }

  void zero_grad() noexcept override {
    for (auto &m : m_Modules)
      m->zero_grad();
  }

private:
  std::vector<std::unique_ptr<Module<T>>> m_Modules;

  // Rebuilt on each parameters() call - stable as long as no modules are added
  // after training starts
  std::vector<Parameter<T> *> m_Params;
};
} // namespace ml::core
