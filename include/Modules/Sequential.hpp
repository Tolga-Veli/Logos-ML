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

  void add(std::unique_ptr<Module> module);

  template <class ModuleType, class... Args>
    requires std::is_base_of_v<Module, ModuleType>
  void add(Args &&...args) {
    add(std::make_unique<ModuleType>(std::forward<Args>(args)...));
  }

  Tensor forward(const Tensor &X) override;
  Tensor backward(const Tensor &Y) override;

  [[nodiscard]] Shape output_shape(const Shape &in) const override;
  [[nodiscard]] Shape input_shape(const Shape &out) const override;

private:
  std::vector<std::unique_ptr<Module>> m_Modules;
};
} // namespace ml::core
