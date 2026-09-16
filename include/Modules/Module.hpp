#pragma once

#include "Core/Tensor.hpp"
#include "Parameter.hpp"

#include <span>
#include <vector>

namespace ml::core {

class Module {
public:
  virtual ~Module() = default;

  virtual void forward(const Tensor &X, Tensor &Y) = 0;
  virtual void backward(const Tensor &Y, Tensor &X) = 0;

  virtual std::span<Parameter *const> own_parameters() { return {}; }

  // Given an input shape, what shape does forward() produce?
  // Default: same shape (correct for ReLU, and any elementwise op).
  virtual Shape output_shape(const Shape &in) const { return in; }

  // Given an output-gradient shape, what shape does backward() produce?
  // Default: same shape
  virtual Shape input_shape(const Shape &out) const { return out; }

  [[nodiscard]] std::vector<Parameter *> parameters();

  void zero_grad() noexcept;

  // Train / eval mode
  // Propagates recursively to all children
  bool training = true;

  void train(bool mode = true);
  void eval() { train(false); }

protected:
  // Child module registration
  // register_module(child) in your constructor for each child_module
  void register_module(Module &child) { m_Children.push_back(&child); }

private:
  std::vector<Module *> m_Children;
};
} // namespace ml::core
