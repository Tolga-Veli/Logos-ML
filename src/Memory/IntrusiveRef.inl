#pragma once

#include "Memory/IntrusiveRef.hpp"

#include <utility>

namespace ml::memory {

template <IntrusiveRefBase T> IntrusiveRef<T>::IntrusiveRef(T *ptr) noexcept : m_Ptr(ptr) {
  if (m_Ptr)
    m_Ptr->add_ref();
}

template <IntrusiveRefBase T> IntrusiveRef<T>::IntrusiveRef(const IntrusiveRef &other) noexcept : m_Ptr(other.m_Ptr) {
  if (m_Ptr)
    m_Ptr->add_ref();
}

template <IntrusiveRefBase T> IntrusiveRef<T> &IntrusiveRef<T>::operator=(const IntrusiveRef &other) noexcept {
  if (this == &other)
    return *this;

  if (other.m_Ptr)
    other.m_Ptr->add_ref();

  release();

  m_Ptr = other.m_Ptr;
  return *this;
}

template <IntrusiveRefBase T> IntrusiveRef<T>::IntrusiveRef(IntrusiveRef &&other) noexcept
    : m_Ptr(std::exchange(other.m_Ptr, nullptr)) {}

template <IntrusiveRefBase T> IntrusiveRef<T> &IntrusiveRef<T>::operator=(IntrusiveRef &&other) noexcept {
  if (this == &other)
    return *this;

  release();

  m_Ptr = std::exchange(other.m_Ptr, nullptr);
  return *this;
}

template <IntrusiveRefBase T> void IntrusiveRef<T>::reset() noexcept {
  release();
  m_Ptr = nullptr;
}

template <IntrusiveRefBase T> void IntrusiveRef<T>::release() noexcept {
  if (m_Ptr && m_Ptr->release_ref())
    delete m_Ptr;
}

} // namespace ml::memory
