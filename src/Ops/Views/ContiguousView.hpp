#pragma once

#include "Core/Tensor.hpp"
#include "Utils.hpp"

#include <cstddef>
#include <type_traits>

namespace ml::backend {

template <ViewBaseType T> class ContiguousView {
public:
  using value_type = std::remove_const_t<T>;

  ContiguousView(core::Tensor &tensor)
    requires(!std::is_const_v<T>)
  {
    if (!tensor.is_contiguous())
      throw std::invalid_argument("ContiguousView: Tensor must be contiguous");

    m_Data = tensor.data<value_type>();
    m_Size = tensor.num_elements();
  }

  ContiguousView(const core::Tensor &tensor)
    requires std::is_const_v<T>
  {
    if (!tensor.is_contiguous())
      throw std::invalid_argument("ContiguousView: Tensor must be contiguous");

    m_Data = tensor.data<value_type>();
    m_Size = tensor.num_elements();
  }

  ContiguousView(core::Tensor &&) = delete;
  ContiguousView(const core::Tensor &&) = delete;

  template <ViewBaseType U>
    requires(std::is_const_v<T> && std::is_same_v<U, value_type>)
  ContiguousView(const ContiguousView<U> &other) noexcept : m_Data(other.data()), m_Size(other.size()) {}

  [[nodiscard]] T *data() const noexcept { return m_Data; }
  [[nodiscard]] std::size_t size() const noexcept { return m_Size; }
  [[nodiscard]] T &operator[](std::size_t i) const noexcept {
    CORE_ASSERT(i < m_Size, "ContiguousView: Index out of bounds");
    return m_Data[i];
  }

private:
  T *m_Data{};
  std::size_t m_Size{};
};
} // namespace ml::backend
