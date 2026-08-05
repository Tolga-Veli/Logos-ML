#pragma once

#include "Core/Tensor.hpp"
#include "Parameter.hpp"

#include <span>

namespace ml::core {

template <class T> class Module {
public:
  using Tensor = Tensor<T>;

  virtual ~Module() = default;

  // Takes a batch of inputs, returns a batch of outputs
  virtual Tensor forward(const Tensor &input) = 0;

  // Takes the upstream gradient (same shape as forward's output)
  // returns the gradient w.r.t. the input (same shape as forward's input)
  // No backprop yet
  virtual Tensor backward(const Tensor &grad_out) = 0;

  virtual void zero_grad() noexcept = 0;

  // Returns pointers to all learnable paramters
  virtual std::span<Parameter<T> *const> parameters() = 0;
};
} // namespace ml::core
