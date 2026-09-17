#pragma once

#include "TensorImpl.hpp"

namespace ml::core {
template <class T, Index... Indices> T &TensorImpl::at(Indices... indices) {
  const std::array<std::size_t, sizeof...(Indices)> idx{static_cast<std::size_t>(indices)...};
  return data<T>()[ComputeStorageOffset(idx)];
}

template <class T, Index... Indices> const T &TensorImpl::at(Indices... indices) const {
  const std::array<std::size_t, sizeof...(Indices)> idx{static_cast<std::size_t>(indices)...};
  return data<T>()[ComputeStorageOffset(idx)];
}

// Deep copy:
template <class Self> memory::IntrusiveRef<Self> TensorImpl::clone() const {
  auto out = ml::memory::CreateIntrusiveRef<Self>(m_Shape, m_Dtype, m_Storage->device());
  CopyElementsInto(*out);
  return out;
}
} // namespace ml::core
