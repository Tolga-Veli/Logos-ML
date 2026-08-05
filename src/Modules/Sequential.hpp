#pragma once

#include "Module.hpp"

#include <memory>
#include <type_traits>
#include <vector>

namespace ml::core {
template <class T> class Sequential final : public Module<T> {
public:
  Sequential() = default;

  void add(std::unique_ptr<Module<T>> module) {
    this->register_module(*module);
    m_Modules.push_back(std::move(module));
  }

  template <class ModuleType, class... Args>
    requires std::is_base_of_v<Module<T>, ModuleType>
  void add(Args &&...args) {
    add(std::make_unique<ModuleType>(std::forward<Args>(args)...));
  }

  Tensor<T> forward(const Tensor<T> &input) override {
    Tensor tmp1 = input, tmp2;
    for (auto &m : m_Modules) {
      tmp2 = m->forward(tmp1);
      std::swap(tmp1, tmp2);
    }
    return tmp1;
  }

  Tensor<T> backward(const Tensor<T> &grad_out) override {
    Tensor<T> grad = grad_out;
    for (auto it = m_Modules.rbegin(); it != m_Modules.rend(); ++it)
      grad = (*it)->backward(grad);
    return grad;
  }

private:
  std::vector<std::unique_ptr<Module<T>>> m_Modules;
};
} // namespace ml::core
