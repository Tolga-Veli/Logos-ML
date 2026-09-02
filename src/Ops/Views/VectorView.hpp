#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"

#include <cstddef>
#include <type_traits>

namespace ml {
namespace kernels {

template <class T> struct VectorView {
  explicit VectorView(core::Tensor &tensor)
      : m_Data(tensor.data<T>()), m_Size(tensor.shape()[0]),
        m_Stride(tensor.strides()[0]) {
    CORE_VERIFY(tensor.rank() == 1, "VectorView<T,1>: tensor must be rank-1");
  }

  explicit VectorView(const core::Tensor &tensor)
    requires std::is_const_v<T>
      : m_Data(tensor.data<std::remove_const_t<T>>()),
        m_Size(tensor.shape()[0]), m_Stride(tensor.strides()[0]) {
    CORE_VERIFY(tensor.rank() == 1, "VectorView<T,1>: tensor must be rank-1");
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

} // namespace kernels

namespace ops {
using kernels::VectorView;
}
} // namespace ml
