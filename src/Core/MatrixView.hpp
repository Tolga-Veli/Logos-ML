#pragma once

#include "Core/Tensor.hpp"
#include <cstdlib>

namespace ml::core {

// a thin-wrapper used for linear algebra operations (matmul)
template <class T> struct MatrixView {
  explicit MatrixView(Tensor<T> &tensor)
      : m_Data(tensor.data()), m_Rows(tensor.shape()[0]),
        m_Cols(tensor.shape()[1]), m_LeadingDim(tensor.strides()[0]) {
    assert(tensor.rank() == 2);
    assert(tensor.strides()[1] == 1);
  }

  explicit MatrixView(const Tensor<std::remove_const_t<T>> &tensor)
      : m_Data(tensor.data()), m_Rows(tensor.shape()[0]),
        m_Cols(tensor.shape()[1]), m_LeadingDim(tensor.strides()[0]) {
    assert(tensor.rank() == 2);
    assert(tensor.strides()[1] == 1);
  }

  [[nodiscard]] T *data() noexcept { return m_Data; }
  [[nodiscard]] const T *data() const noexcept { return m_Data; }
  [[nodiscard]] int rows() const noexcept { return m_Rows; }
  [[nodiscard]] int cols() const noexcept { return m_Cols; }
  [[nodiscard]] int ld() const noexcept { return m_LeadingDim; }

private:
  T *m_Data;
  int m_Rows, m_Cols, m_LeadingDim;
};

template <class T> inline MatrixView<T> as_matrix(Tensor<T> &t) {
  assert(t.rank() == 2);
  return {t.data(), t.shape()[0], t.shape()[1]};
}

template <class T> inline MatrixView<const T> as_matrix(const Tensor<T> &t) {
  assert(t.rank() == 2);
  return {t.data(), t.shape()[0], t.shape()[1]};
}

} // namespace ml::core
