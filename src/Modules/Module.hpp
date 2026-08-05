#pragma once

#include "Core/Tensor.hpp"
#include "Parameter.hpp"

#include <span>
#include <vector>

namespace ml::core {

template <class T> class Module {
public:
  using Scalar = T;

  virtual ~Module() = default;

  virtual Tensor<T> forward(const Tensor<T> &input) = 0;
  virtual Tensor<T> backward(const Tensor<T> &grad_out) = 0;

  virtual std::span<Parameter<T> *const> own_parameters() { return {}; }

  std::vector<Parameter<T> *> parameters() {
    std::vector<Parameter<T> *> params;

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

  // Train / eval mode
  // Propagates recursively to all children
  bool training = true;

  void train(bool mode = true) {
    training = mode;
    for (auto *child : m_Children)
      child->train(mode);
  }

  void eval() { train(false); }

protected:
  // Child module registration
  // register_module(child) in your constructor for each child_module
  void register_module(Module<T> &child) { m_Children.push_back(&child); }

private:
  std::vector<Module<T> *> m_Children;
};
} // namespace ml::core
