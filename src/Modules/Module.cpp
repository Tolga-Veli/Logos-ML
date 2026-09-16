#include "Modules/Module.hpp"

namespace ml::core {
std::vector<Parameter *> Module::parameters() {
  std::vector<Parameter *> params;

  for (auto *p : own_parameters())
    params.push_back(p);

  for (auto *child : m_Children)
    for (auto *p : child->parameters())
      params.push_back(p);

  return params;
}

void Module::zero_grad() noexcept {
  for (auto *p : own_parameters())
    p->zero_grad();

  for (auto *child : m_Children)
    child->zero_grad();
}

void Module::train(bool mode) {
  training = mode;
  for (auto *child : m_Children)
    child->train(mode);
}

} // namespace ml::core
