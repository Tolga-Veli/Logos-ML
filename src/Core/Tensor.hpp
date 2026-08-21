#pragma once

#include "Core/DType.hpp"
#include "Memory/Device.hpp"
#include "Memory/IntrusiveRef.hpp"
#include "TensorImpl.hpp"

#include <cassert>

namespace ml::core {

// Tensor is a thin, cheaply-copyable handle around an intrusively refcounted
// TensorImpl
class Tensor {
public:
  Tensor() = default;
  explicit Tensor(const Shape &shape, DType type = DType::Float32)
      : m_Impl(memory::CreateIntrusiveRef<TensorImpl>(shape, type)) {}

  Tensor(const Tensor &) = default;
  Tensor &operator=(const Tensor &) = default;
  Tensor(Tensor &&) noexcept = default;
  Tensor &operator=(Tensor &&) noexcept = default;
  ~Tensor() = default;

  [[nodiscard]] int rank() const noexcept { return m_Impl->rank(); }
  [[nodiscard]] int num_elements() const noexcept {
    return m_Impl->num_elements();
  }
  [[nodiscard]] DType dtype() const noexcept { return m_Impl->dtype(); }

  [[nodiscard]] memory::Device device() const noexcept {
    return m_Impl->device();
  }

  template <class T> [[nodiscard]] T *data() noexcept {
    return m_Impl->data<T>();
  }
  template <class T> [[nodiscard]] const T *data() const noexcept {
    return m_Impl->data<T>();
  }

  std::byte *raw_data() noexcept { return m_Impl->raw_data(); }
  const std::byte *raw_data() const noexcept { return m_Impl->raw_data(); }

  [[nodiscard]] int offset() const noexcept { return m_Impl->offset(); }

  [[nodiscard]]
  const Shape &shape() const noexcept {
    return m_Impl->shape();
  }

  [[nodiscard]]
  const Strides &strides() const noexcept {
    return m_Impl->strides();
  }

  void fill_zero() {
    using enum DType;
    switch (dtype()) {
    case Float32:
      m_Impl->fill<float>(0.0f);
      break;
    case Float64:
      m_Impl->fill<double>(0.0f);
      break;
    case Int32:
      m_Impl->fill<int>(0.0f);
      break;
    }
  }

  [[nodiscard]]
  bool is_contiguous() const noexcept {
    return m_Impl->is_contiguous();
  }

  // Deep copy
  [[nodiscard]]
  Tensor clone() const {
    return Tensor(m_Impl->clone());
  }

  // T(1,2,3,...) access
  template <class T, typename... Indices>
    requires(std::convertible_to<Indices, int> && ...)
  T &operator()(Indices... indices) {
    return m_Impl->template operator()<T>(indices...);
  }

  template <class T, typename... Indices>
    requires(std::convertible_to<Indices, int> && ...)
  const T &operator()(Indices... indices) const {
    return m_Impl->template operator()<T>(indices...);
  }

private:
  memory::IntrusiveRef<TensorImpl> m_Impl;

  explicit Tensor(memory::IntrusiveRef<TensorImpl> impl)
      : m_Impl(std::move(impl)) {}
};
} // namespace ml::core
