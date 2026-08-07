#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"
#include "TensorView.hpp"

#include <cstdlib>
#include <type_traits>

namespace ml {
namespace kernels {
// ---------------------------------------------------------------------
// Rank-0 specialization: ScalarView - a view over a single element.
// ---------------------------------------------------------------------
template <class T> struct TensorView<T, 0> {
  static constexpr std::size_t Rank = 0;

  explicit TensorView(Tensor &tensor) : m_Data(tensor.data<T>()) {
    CORE_VERIFY(tensor.rank() == 0, "ScalarView: tensor must be rank-0");
  }

  explicit TensorView(const Tensor &tensor)
    requires std::is_const_v<T>
      : m_Data(tensor.data<std::remove_const_t<T>>()) {
    CORE_VERIFY(tensor.rank() == 0, "ScalarView: tensor must be rank-0");
  }

  [[nodiscard]] T *data() noexcept { return m_Data; }
  [[nodiscard]] const T *data() const noexcept { return m_Data; }

  [[nodiscard]] T &value() const noexcept { return *m_Data; }

private:
  T *m_Data;
};

template <class T> using ScalarView = TensorView<T, 0>;
} // namespace kernels

namespace ops {
using kernels::ScalarView;
}
} // namespace ml
