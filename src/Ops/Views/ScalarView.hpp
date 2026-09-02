#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"

#include <type_traits>

namespace ml {
namespace kernels {
template <class T> struct ScalarView {
  explicit ScalarView(core::Tensor &tensor) : m_Data(tensor.data<T>()) {
    CORE_VERIFY(tensor.rank() == 0, "ScalarView: tensor must be rank-0");
  }

  explicit ScalarView(const core::Tensor &tensor)
    requires std::is_const_v<T>
      : m_Data(tensor.data<std::remove_const_t<T>>()) {
    CORE_VERIFY(tensor.rank() == 0, "ScalarView: tensor must be rank-0");
  }

  [[nodiscard]] T *data() noexcept { return m_Data; }
  [[nodiscard]] const T *data() const noexcept { return m_Data; }

  [[nodiscard]] T &value() noexcept
    requires(!std::is_const_v<T>)
  {
    return *m_Data;
  }

private:
  T *m_Data;
};
} // namespace kernels

namespace ops {
using kernels::ScalarView;
}
} // namespace ml
