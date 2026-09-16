#pragma once

#include "Core/Assert.hpp"

#include "Ops/Views/ContiguousView.hpp"

#include <algorithm>
#include <cmath>

namespace ml::backend::cpu {

template <class T> void relu(ContiguousView<const T> in, ContiguousView<T> out) {
  CORE_ASSERT(in.size() == out.size(), "Shape mismatch");

  for (std::size_t i = 0; i < in.size(); i++)
    out[i] = std::max(in[i], T{0});
}

template <class T>
void relu_backward(ContiguousView<const T> grad_out, ContiguousView<const T> x, ContiguousView<T> grad_in) {
  CORE_VERIFY(grad_out.size() == x.size() && grad_in.size() == x.size(), "Shape mismatch");

  for (std::size_t i = 0; i < x.size(); i++)
    grad_in[i] = x[i] > T{0} ? grad_out[i] : T{0};
}
} // namespace ml::backend::cpu
