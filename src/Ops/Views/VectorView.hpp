#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"
#include "TensorView.hpp"

#include <cstdlib>
#include <type_traits>

namespace ml {
namespace kernels {
// ---------------------------------------------------------------------
// Rank-1 specialization: VectorView - strided 1D view (dot, axpy, gemv).
// ---------------------------------------------------------------------
template <class T> struct TensorView<T, 1> {
  using ElementType = T;
  static constexpr std::size_t Rank = 1;

  explicit TensorView(Tensor &tensor)
      : m_Data(tensor.data<T>()), m_Size(tensor.shape()[0]),
        m_Stride(tensor.strides()[0]) {
    CORE_VERIFY(tensor.rank() == 1, "TensorView<T,1>: tensor must be rank-1");
  }

  explicit TensorView(const Tensor &tensor)
    requires std::is_const_v<T>
      : m_Data(tensor.data<std::remove_const_t<T>>()),
        m_Size(tensor.shape()[0]), m_Stride(tensor.strides()[0]) {
    CORE_VERIFY(tensor.rank() == 1, "TensorView<T,1>: tensor must be rank-1");
  }

  [[nodiscard]] T *data() noexcept { return m_Data; }
  [[nodiscard]] const T *data() const noexcept { return m_Data; }
  [[nodiscard]] int size() const noexcept { return m_Size; }
  [[nodiscard]] int stride() const noexcept { return m_Stride; }
  // BLAS-style alias: many BLAS APIs call this "incx"/"incy".
  [[nodiscard]] int inc() const noexcept { return m_Stride; }

  [[nodiscard]] T &operator()(std::size_t i) const noexcept {
    return m_Data[i * static_cast<std::size_t>(m_Stride)];
  }
  [[nodiscard]] T &operator[](std::size_t i) const noexcept {
    return m_Data[i * static_cast<std::size_t>(m_Stride)];
  }

private:
  T *m_Data;
  int m_Size, m_Stride;
};

template <class T> using VectorView = TensorView<T, 1>;

} // namespace kernels

namespace ops {
using kernels::VectorView;
}
} // namespace ml
