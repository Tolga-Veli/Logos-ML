#pragma once

#include "TensorImpl.hpp"

namespace ml::core {
template <class T, Index... Indices> T &TensorImpl::operator()(Indices... indices) {
  std::array<int, sizeof...(Indices)> idx{indices...};
  return data<T>()[ComputeStorageOffset(idx)];
}

template <class T, Index... Indices> const T &TensorImpl::operator()(Indices... indices) const {
  std::array<int, sizeof...(Indices)> idx{indices...};
  return data<T>()[ComputeStorageOffset(idx)];
}

// Deep copy:
template <class Self> memory::IntrusiveRef<Self> TensorImpl::clone() const {
  auto out = ml::memory::CreateIntrusiveRef<Self>(m_Shape, m_Dtype);
  CopyElementsInto(*out);
  return out;
}
} // namespace ml::core
