#include "Modules/Module.hpp"

#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace ml::core {

void Module::validate_name(std::string_view name) const {
  if (name.empty())
    throw std::invalid_argument("Module: registration name must not be empty");

  for (const auto &entry : m_Parameters)
    if (entry.name == name)
      throw std::invalid_argument("Module: duplicate registration name '" + std::string(name) + "'");

  for (const auto &entry : m_Children)
    if (entry.name == name)
      throw std::invalid_argument("Module: duplicate registration name '" + std::string(name) + "'");
}

void Module::register_parameter(std::string name, Parameter &parameter) {
  validate_name(name);
  m_Parameters.push_back({std::move(name), &parameter});
}

Module &Module::register_module(std::string name, std::unique_ptr<Module> child) {
  if (!child)
    throw std::invalid_argument("Module: child must not be null");

  validate_name(name);
  child->train(m_Training);
  m_Children.push_back({std::move(name), std::move(child)});
  return *m_Children.back().module;
}

std::vector<Parameter *> Module::parameters() {
  std::vector<Parameter *> res;
  std::unordered_set<const Parameter *> seen;

  const auto get = [&](auto &&self, Module &module) -> void {
    for (const auto &entry : module.m_Parameters)
      if (seen.insert(entry.parameter).second)
        res.push_back(entry.parameter);

    for (const auto &entry : module.m_Children)
      self(self, *entry.module);
  };

  get(get, *this);
  return res;
}

std::vector<const Parameter *> Module::parameters() const {
  std::vector<const Parameter *> res;
  std::unordered_set<const Parameter *> seen;

  const auto get = [&](auto &&self, const Module &module) -> void {
    for (const auto &entry : module.m_Parameters)
      if (seen.insert(entry.parameter).second)
        res.push_back(entry.parameter);

    for (const auto &entry : module.m_Children)
      self(self, *entry.module);
  };

  get(get, *this);
  return res;
}

void Module::zero_grad() {
  for (auto *parameter : parameters())
    parameter->zero_grad();
}

void Module::train(bool mode) {
  m_Training = mode;
  for (const auto &entry : m_Children)
    entry.module->train(mode);
}

} // namespace ml::core
