#pragma once

#include "Core/Tensor.hpp"
#include <cstdlib>

namespace ml::core {

// a thin-wrapper used for linear algebra operations (matmul)
template <class T> struct MatrixView {
  explicit MatrixView(const Tensor<T> &tensor)
      : m_Data(tensor.data()), m_Rows(tensor.GetShape()[0]),
        m_Cols(tensor.GetShape()[1]), m_LeadingDim(tensor.GetStrides()[0]) {
    assert(tensor.rank() == 2);
    assert(tensor.GetStrides()[1] == 1);
  }

  [[nodiscard]] T *data() { return m_Data; }
  [[nodiscard]] const T *data() const { return m_Data; }

  int rows() const { return m_Rows; }
  int cols() const { return m_Cols; }
  int ld() const { return m_LeadingDim; }

private:
  T *m_Data;
  int m_Rows, m_Cols, m_LeadingDim;
};
} // namespace ml::core
