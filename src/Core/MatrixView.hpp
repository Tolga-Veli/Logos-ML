#pragma once

#include "Core/Tensor.hpp"
#include <cstdlib>

namespace ml::core {

// a thin-wrapper used for linear algebra operations (matmul)
template <class T> struct MatrixView {
  MatrixView(Tensor<T> &tensor) { assert(tensor.rank() == 2); }

  T *data() { return m_Data; }
  const T *data() const { return m_Data; }

  std::size_t rows() const { return m_Rows; }
  std::size_t cols() const { return m_Cols; }
  std::size_t row_stride() const { return m_RowStride; }
  std::size_t col_stride() const { return m_ColStride; }

  bool is_row_major() const {
    return (m_RowStride == m_Cols && m_ColStride == 1);
  }

  bool is_col_major() const {
    return (m_RowStride == 1 && m_ColStride == m_Rows);
  }

  T &operator()(std::size_t i, std::size_t j) const {
    return m_Data[i * m_RowStride + j * m_ColStride];
  }

private:
  T *m_Data;

  std::size_t m_Rows, m_Cols;
  std::size_t m_RowStride, m_ColStride;
};
} // namespace ml::core
