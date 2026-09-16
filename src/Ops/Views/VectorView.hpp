#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"
#include "Utils.hpp"

#include <cstddef>
#include <type_traits>

namespace ml::backend {

template <ViewBaseType T> class VectorView {
public:
  using value_type = std::remove_const_t<T>;

  VectorView(core::Tensor &tensor)
    requires(!std::is_const_v<T>)
  {
    CORE_ASSERT(tensor.rank() == 1, "Tensor must be rank-1");

    m_Data = tensor.data<value_type>();
    m_Size = tensor.shape()[0];
    m_Stride = tensor.strides()[0];
  }

  VectorView(const core::Tensor &tensor)
    requires std::is_const_v<T>
  {
    CORE_ASSERT(tensor.rank() == 1, "Tensor must be rank-1");

    m_Data = tensor.data<value_type>();
    m_Size = tensor.shape()[0];
    m_Stride = tensor.strides()[0];
  }

  VectorView(core::Tensor &&) = delete;
  VectorView(const core::Tensor &&) = delete;

  template <ViewBaseType U>
    requires(std::is_const_v<T> && std::is_same_v<U, value_type>)
  constexpr VectorView(const VectorView<U> &other) noexcept
      : m_Data(other.data()), m_Size(other.size()), m_Stride(other.stride()) {}

  [[nodiscard]] T *data() const noexcept { return m_Data; }
  [[nodiscard]] std::size_t size() const noexcept { return m_Size; }
  [[nodiscard]] std::size_t stride() const noexcept { return m_Stride; }
  [[nodiscard]] bool empty() const noexcept { return m_Size == 0; }

  [[nodiscard]] bool is_contiguous() const noexcept { return m_Size <= 1 || m_Stride == 1; }

  [[nodiscard]] T &operator[](std::size_t i) const noexcept {
    CORE_ASSERT(i < m_Size, "Index out of bounds");
    return m_Data[i * m_Stride];
  }

private:
  T *m_Data{};
  std::size_t m_Size{}, m_Stride{};
};
} // namespace ml::backend
