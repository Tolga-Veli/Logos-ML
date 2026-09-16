#pragma once

#include "Core/Error.hpp"
#include "Core/Tensor.hpp"

#include "Utils.hpp"

#include <type_traits>

namespace ml::backend {
template <ViewBaseType T> class ScalarView {
public:
  using value_type = std::remove_const_t<T>;

  ScalarView(core::Tensor &tensor)
    requires(!std::is_const_v<T>)
  {
    if (tensor.rank() != 0)
      throw ShapeError("ScalarView: Tensor must be rank zero");

    m_Data = tensor.data<value_type>();
  }

  ScalarView(const core::Tensor &tensor)
    requires std::is_const_v<T>
  {
    if (tensor.rank() != 0)
      throw ShapeError("ScalarView: Tensor must be rank zero");

    m_Data = tensor.data<value_type>();
  }

  ScalarView(core::Tensor &&) = delete;
  ScalarView(const core::Tensor &&) = delete;

  template <class U>
    requires(std::is_const_v<T> && std::is_same_v<U, value_type>)
  constexpr ScalarView(const ScalarView<U> &other) noexcept : m_Data(other.data()) {}

  [[nodiscard]] constexpr T *data() const noexcept { return m_Data; }
  [[nodiscard]] constexpr T &value() const noexcept { return *m_Data; }

private:
  T *m_Data{};
};
} // namespace ml::backend
