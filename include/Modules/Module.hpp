#pragma once

#include "Core/Tensor.hpp"
#include "Parameter.hpp"

#include <span>
#include <vector>

namespace ml::core {

class Module {
public:
  virtual ~Module() = default;

  virtual Tensor forward(const Tensor &X) = 0;
  virtual Tensor backward(const Tensor &Y) = 0;

  virtual std::span<Parameter *const> own_parameters() { return {}; }

  // Given an output-gradient shape, what shape does backward() produce?
  // Default: same shape
  virtual Shape input_shape(const Shape &out) const { return out; }

  // Given an input shape, what shape does forward() produce?
  // Default: same shape (correct for ReLU, and any elementwise op).
  virtual Shape output_shape(const Shape &in) const { return in; }

  std::vector<Parameter *> parameters() {
    std::vector<Parameter *> params;

    for (auto *p : own_parameters())
      params.push_back(p);

    for (auto *child : m_Children)
      for (auto *p : child->parameters())
        params.push_back(p);

    return params;
  }

  void zero_grad() noexcept {
    for (auto *p : own_parameters())
      p->zero_grad();

    for (auto *child : m_Children)
      child->zero_grad();
  }

  void train(bool mode = true) {
    m_Training = mode;
    for (auto *child : m_Children)
      child->train(mode);
  }

  void eval() { train(false); }

protected:
  // Child module registration
  // register_module(child) in your constructor for each child_module
  void register_module(Module &child) { m_Children.push_back(&child); }

private:
  std::vector<Module *> m_Children;
  bool m_Training{true};
};
} // namespace ml::core
