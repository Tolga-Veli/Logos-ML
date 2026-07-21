#pragma once

#include "Core/Tensor.hpp"
#include <cassert>
#include <cstddef>

namespace ml::core {

template <class T> class VectorView {
public:
  // Built from a rank-1 Tensor. `inc` is BLAS's "increment" -- how many
  // elements to skip between logical entries. For a Tensor straight out
  // of the current API this is always 1 (rank-1 tensors are always
  // contiguous, same invariant as Tensor's last dimension), but the
  // field exists so a future strided/column view can populate it.
  explicit VectorView(Tensor<T> &t)
      : m_Data(t.data()), m_Size(t.GetShape()[0]), m_Inc(t.GetStrides()[0]) {
    assert(t.rank() == 1);
  }

  explicit VectorView(const Tensor<T> &t)
      : m_Data(t.data()), m_Size(t.GetShape()[0]), m_Inc(t.GetStrides()[0]) {
    assert(t.rank() == 1);
  }

  [[nodiscard]] T *data() const { return m_Data; }
  [[nodiscard]] size_t size() const { return m_Size; }
  [[nodiscard]] size_t inc() const { return m_Inc; }

private:
  T *m_Data;
  size_t m_Size, m_Inc;
};

} // namespace ml::core
