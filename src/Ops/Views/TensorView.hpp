#pragma once

#include "Core/Assert.hpp"
#include "Core/Tensor.hpp"

#include <array>
#include <cstdlib>
#include <type_traits>

namespace ml {
namespace kernels {
using core::Tensor;
/*
 * Generic N-rank strided view over a Tensor's storage
 *  T - element type (const T for read-only views)
 *  N - rank
 */
template <class T, std::size_t N> struct TensorView {
  static constexpr std::size_t rank = N;

  explicit TensorView(Tensor &tensor)
      : m_Data(tensor.data<T>()), m_Shape(tensor.shape()),
        m_Strides(tensor.strides()) {
    CORE_VERIFY(tensor.rank() == N, "TensorView: rank must match");
  }

  explicit TensorView(const Tensor &tensor)
    requires std::is_const_v<T>
      : m_Data(tensor.data<std::remove_const_t<T>>()), m_Shape(tensor.shape()),
        m_Strides(tensor.strides()) {
    CORE_VERIFY(tensor.rank() == N, "TensorView: rank must match");
  }

  [[nodiscard]] T *data() noexcept { return m_Data; }
  [[nodiscard]] const T *data() const noexcept { return m_Data; }

  [[nodiscard]] std::size_t size(std::size_t dim) const noexcept {
    return m_Shape[dim];
  }

  [[nodiscard]] std::size_t stride(std::size_t dim) const noexcept {
    return m_Strides[dim];
  }

  [[nodiscard]]
  const std::array<std::size_t, N> &shape() const noexcept {
    return m_Shape;
  }

  [[nodiscard]]
  const std::array<std::size_t, N> &strides() const noexcept {
    return m_Strides;
  }

  // Element access: v(i, j, k, ...) - one index per dimension.
  template <class... Idx>
    requires(sizeof...(Idx) == N)
  [[nodiscard]] T &operator()(Idx... idx) const noexcept {
    std::size_t indices[N] = {static_cast<std::size_t>(idx)...}, offset = 0;
    for (std::size_t i{0}; i < N; i++)
      offset += indices[i] * m_Strides[i];
    return m_Data[offset];
  }

private:
  T *m_Data;
  std::array<std::size_t, N> m_Shape{}, m_Strides{};
};
} // namespace kernels

namespace ops {
using kernels::TensorView;
} // namespace ops
} // namespace ml
