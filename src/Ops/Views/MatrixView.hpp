#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"
#include "Utils.hpp"

#include <cstddef>
#include <type_traits>

namespace ml::backend {

template <ViewBaseType T> class MatrixView {
public:
  using value_type = std::remove_const_t<T>;

  MatrixView(core::Tensor &tensor)
    requires(!std::is_const_v<T>)
  {
    CORE_ASSERT(tensor.rank() == 2, "Tensor must be rank-2");

    m_Data = tensor.data<value_type>();
    m_Rows = tensor.shape()[0];
    m_Cols = tensor.shape()[1];
    m_RowStride = tensor.strides()[0];
    m_ColStride = tensor.strides()[1];
  }

  MatrixView(const core::Tensor &tensor)
    requires std::is_const_v<T>
  {
    CORE_VERIFY(tensor.rank() == 2, "Tensor must be rank-2");

    m_Data = tensor.data<value_type>();
    m_Rows = tensor.shape()[0];
    m_Cols = tensor.shape()[1];
    m_RowStride = tensor.strides()[0];
    m_ColStride = tensor.strides()[1];
  }

  MatrixView(core::Tensor &&) = delete;
  MatrixView(const core::Tensor &&) = delete;

  template <ViewBaseType U>
    requires(std::is_const_v<T> && std::is_same_v<U, value_type>)
  MatrixView(const MatrixView<U> &other) noexcept
      : m_Data(other.data()), m_Rows(other.rows()), m_Cols(other.cols()), m_RowStride(other.row_stride()),
        m_ColStride(other.col_stride()) {}

  [[nodiscard]] T *data() const noexcept { return m_Data; }

  [[nodiscard]] std::size_t rows() const noexcept { return m_Rows; }
  [[nodiscard]] std::size_t cols() const noexcept { return m_Cols; }
  [[nodiscard]] std::size_t row_stride() const noexcept { return m_RowStride; }
  [[nodiscard]] std::size_t col_stride() const noexcept { return m_ColStride; }

  [[nodiscard]] bool is_row_major() const noexcept {
    return ((m_Cols <= 1 || m_ColStride == 1) && (m_Rows <= 1 || m_RowStride == m_Cols));
  }
  [[nodiscard]] bool is_col_major() const noexcept {
    return ((m_Rows <= 1 || m_RowStride == 1) && (m_Cols <= 1 || m_ColStride == m_Rows));
  }

  [[nodiscard]] bool is_contiguous() const noexcept { return is_row_major() || is_col_major(); }

  [[nodiscard]] T &operator()(std::size_t i, std::size_t j) const noexcept {
    CORE_ASSERT(i < m_Rows && j < m_Cols, "Index out of bounds");
    return m_Data[i * m_RowStride + j * m_ColStride];
  }

private:
  T *m_Data{};
  std::size_t m_Rows{}, m_Cols{}, m_RowStride{}, m_ColStride{};
};
} // namespace ml::backend
