#pragma once

#include "Core/Tensor.hpp"
#include <cassert>

namespace ml::core {

template <class T> class VectorView {
public:
  explicit VectorView(Tensor<T> &tensor)
      : m_Data(tensor.data()), m_Size(tensor.shape()[0]),
        m_Inc(tensor.strides()[0]) {
    assert(tensor.rank() == 1);
  }

  explicit VectorView(const Tensor<std::remove_const_t<T>> &tensor)
      : m_Data(tensor.data()), m_Size(tensor.shape()[0]),
        m_Inc(tensor.strides()[0]) {
    assert(tensor.rank() == 1);
  }

  [[nodiscard]] T *data() const { return m_Data; }
  [[nodiscard]] int size() const { return m_Size; }
  [[nodiscard]] int inc() const { return m_Inc; }

private:
  T *m_Data;
  int m_Size, m_Inc;
};

template <class T> VectorView<T> as_vector(Tensor<T> &t) {
  assert(t.rank() == 1);
  return VectorView<T>(t);
}

template <class T> VectorView<const T> as_vector(const Tensor<T> &t) {
  assert(t.rank() == 1);
  return VectorView<const T>(t);
}

} // namespace ml::core
