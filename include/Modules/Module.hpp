#pragma once

#include "Core/Tensor.hpp"
#include "Parameter.hpp"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ml::core {

class Module {
public:
  Module() = default;
  virtual ~Module() = default;

  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;
  Module(Module &&) = delete;
  Module &operator=(Module &&) = delete;

  virtual Tensor forward(const Tensor &X) = 0;
  virtual Tensor backward(const Tensor &Y) = 0;

  // Given an output-gradient shape, what shape does backward() produce?
  // Default: same shape
  virtual Shape input_shape(const Shape &out) const { return out; }

  // Given an input shape, what shape does forward() produce?
  // Default: same shape (correct for ReLU, and any elementwise op).
  virtual Shape output_shape(const Shape &in) const { return in; }

  // Local parameters first, then children in registration order. Shared
  // parameters appear once. Returned pointers do not extend their lifetime.
  [[nodiscard]] std::vector<Parameter *> parameters();
  [[nodiscard]] std::vector<const Parameter *> parameters() const;

  void zero_grad();
  void train(bool mode = true);
  void eval() { train(false); }
  [[nodiscard]] bool is_training() const noexcept { return m_Training; }

protected:
  struct ChildEntry {
    std::string name;
    std::unique_ptr<Module> module;
  };

  // Parameters remain owned by the derived layer and must outlive registry use.
  // Child ownership is transferred to this module. Names share one namespace.
  void register_parameter(std::string name, Parameter &parameter);
  Module &register_module(std::string name, std::unique_ptr<Module> child);

  // Registry entries cannot be replaced through this view. Adding a child can
  // invalidate the view, but does not move the child objects themselves.
  [[nodiscard]] std::span<const ChildEntry> children() const noexcept { return m_Children; }

private:
  struct ParameterEntry {
    std::string name;
    Parameter *parameter;
  };

  void validate_name(std::string_view name) const;

  std::vector<ParameterEntry> m_Parameters;
  std::vector<ChildEntry> m_Children;
  bool m_Training{true};
};
} // namespace ml::core
