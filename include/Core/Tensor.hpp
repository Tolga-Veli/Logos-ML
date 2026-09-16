#pragma once

#include "Core/DType.hpp"
#include "Core/TensorImpl.hpp"
#include "Memory/Device.hpp"
#include "Memory/IntrusiveRef.hpp"

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

  [[nodiscard]] bool is_contiguous() const noexcept { return m_Impl->is_contiguous(); }
  [[nodiscard]] std::size_t rank() const noexcept { return m_Impl->rank(); }
  [[nodiscard]] std::size_t num_elements() const noexcept { return m_Impl->num_elements(); }
  [[nodiscard]] std::size_t offset() const noexcept { return m_Impl->offset(); }
  [[nodiscard]] DType dtype() const noexcept { return m_Impl->dtype(); }
  [[nodiscard]] memory::Device device() const noexcept { return m_Impl->device(); }
  [[nodiscard]] const Shape &shape() const noexcept { return m_Impl->shape(); }
  [[nodiscard]] const Strides &strides() const noexcept { return m_Impl->strides(); }
  [[nodiscard]] Tensor clone() const { return Tensor(m_Impl->clone()); }

  template <class T> [[nodiscard]] T *data() noexcept { return m_Impl->data<T>(); }
  template <class T> [[nodiscard]] const T *data() const noexcept { return m_Impl->data<T>(); }

  std::byte *raw_data() noexcept { return m_Impl->raw_data(); }
  const std::byte *raw_data() const noexcept { return m_Impl->raw_data(); }

  template <class T, typename... Indices>
    requires(std::convertible_to<Indices, std::size_t> && ...)
  T &at(Indices... indices) {
    return m_Impl->at<T>(indices...);
  }

  template <class T, typename... Indices>
    requires(std::convertible_to<Indices, std::size_t> && ...)
  const T &at(Indices... indices) const {
    return m_Impl->at<T>(indices...);
  }

private:
  memory::IntrusiveRef<TensorImpl> m_Impl;

  explicit Tensor(memory::IntrusiveRef<TensorImpl> impl) : m_Impl(std::move(impl)) {}
};
} // namespace ml::core
