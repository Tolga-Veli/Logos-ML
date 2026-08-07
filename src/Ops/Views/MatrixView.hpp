#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"
#include "Ops/Views/TensorView.hpp"

#include <cstdlib>
#include <type_traits>

namespace ml {
namespace kernels {
using core::Tensor;

// ---------------------------------------------------------------------
// Rank-2 specialization: adds the row/col/ld interface your matmul
// kernels want, and enforces row-major (unit inner stride) layout.
// ---------------------------------------------------------------------
template <class T> struct TensorView<T, 2> {
  static constexpr std::size_t Rank = 2;

  explicit TensorView(Tensor &tensor)
      : m_Data(tensor.data<T>()), m_Rows(tensor.shape()[0]),
        m_Cols(tensor.shape()[1]), m_LeadingDim(tensor.strides()[0]) {
    CORE_VERIFY(tensor.rank() == 2, "TensorView<T,2>: tensor must be rank-2");
    CORE_VERIFY(tensor.strides()[1] == 1,
                "TensorView<T,2>: tensor must be row-major");
  }

  explicit TensorView(const Tensor &tensor)
    requires std::is_const_v<T>
      : m_Data(tensor.data<std::remove_const_t<T>>()),
        m_Rows(tensor.shape()[0]), m_Cols(tensor.shape()[1]),
        m_LeadingDim(tensor.strides()[0]) {
    CORE_VERIFY(tensor.rank() == 2, "TensorView<T,2>: tensor must be rank-2");
    CORE_VERIFY(tensor.strides()[1] == 1,
                "TensorView<T,2>: tensor must be row-major");
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

template <class T> using MatrixView = TensorView<T, 2>;

} // namespace kernels

namespace ops {
using kernels::MatrixView;
}
} // namespace ml
