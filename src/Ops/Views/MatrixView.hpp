#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"

#include <cstddef>
#include <type_traits>

namespace ml {
namespace kernels {

template <class T> struct MatrixView {
  explicit MatrixView(core::Tensor &tensor)
      : m_Data(tensor.data<T>()), m_Rows(tensor.shape()[0]),
        m_Cols(tensor.shape()[1]), m_LeadingDim(tensor.strides()[0]) {
    CORE_VERIFY(tensor.rank() == 2, "MatrixView<T,2>: tensor must be rank-2");
    CORE_VERIFY(tensor.strides()[1] == 1,
                "MatrixView<T,2>: tensor must be row-major");
  }

  explicit MatrixView(const core::Tensor &tensor)
    requires std::is_const_v<T>
      : m_Data(tensor.data<std::remove_const_t<T>>()),
        m_Rows(tensor.shape()[0]), m_Cols(tensor.shape()[1]),
        m_LeadingDim(tensor.strides()[0]) {
    CORE_VERIFY(tensor.rank() == 2, "MatrixView<T,2>: tensor must be rank-2");
    CORE_VERIFY(tensor.strides()[1] == 1,
                "MatrixView<T,2>: tensor must be row-major");
  }

  [[nodiscard]] T *data() noexcept { return m_Data; }
  [[nodiscard]] const T *data() const noexcept { return m_Data; }
  [[nodiscard]] int rows() const noexcept { return m_Rows; }
  [[nodiscard]] int cols() const noexcept { return m_Cols; }
  [[nodiscard]] int ld() const noexcept { return m_LeadingDim; }

  [[nodiscard]] T &operator()(std::size_t i, std::size_t j) const noexcept {
    return m_Data[i * static_cast<std::size_t>(m_LeadingDim) + j];
  }

private:
  T *m_Data;
  int m_Rows, m_Cols, m_LeadingDim;
};
} // namespace kernels

namespace ops {
using kernels::MatrixView;
}
} // namespace ml
