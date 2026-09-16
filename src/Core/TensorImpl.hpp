#pragma once

#include "Core/DType.hpp"
#include "Core/Shape.hpp"
#include "Core/Strides.hpp"
#include "Memory/Device.hpp"
#include "Memory/IntrusiveRef.hpp"
#include "Memory/Storage.hpp"

#include <cstring>
#include <span>

namespace ml::core {
template <class T>
concept Index = std::integral<std::remove_cvref_t<T>> && (!std::same_as<std::remove_cvref_t<T>, bool>);

/*
  TensorImpl owns the actual tensor data: storage, shape, strides, offset.
  It is never held by value - always accessed through
  IntrusiveRef<TensorImpl<T>>. Tensor<T> is the thin, cheaply-copyable handle
  around it (see Tensor.hpp)

  Non-copyable, non-movable
*/
class TensorImpl final : public memory::detail::IntrusiveRefCounted {
public:
  explicit TensorImpl(const Shape &shape, DType dtype);

  TensorImpl(memory::IntrusiveRef<memory::Storage> storage, const Shape &shape, const Strides &strides,
             std::size_t offset, DType dtype) noexcept;

  TensorImpl(const TensorImpl &) = delete;
  TensorImpl &operator=(const TensorImpl &) = delete;
  TensorImpl(TensorImpl &&) = delete;
  TensorImpl &operator=(TensorImpl &&) = delete;

  [[nodiscard]] std::size_t rank() const noexcept { return m_Shape.rank(); }
  [[nodiscard]] std::size_t num_elements() const noexcept { return m_Shape.num_elements(); }
  [[nodiscard]] std::size_t offset() const noexcept { return m_Offset; }
  [[nodiscard]] const Shape &shape() const noexcept { return m_Shape; }
  [[nodiscard]] const Strides &strides() const noexcept { return m_Strides; }
  [[nodiscard]] memory::Device device() const noexcept { return m_Storage->device(); }
  [[nodiscard]] DType dtype() const noexcept { return m_Dtype; }
  [[nodiscard]] std::byte *raw_data() noexcept { return m_Storage->raw_data() + m_Offset * dtype_size(m_Dtype); }
  [[nodiscard]] const std::byte *raw_data() const noexcept {
    return m_Storage->raw_data() + m_Offset * dtype_size(m_Dtype);
  }
  [[nodiscard]] bool is_contiguous() const noexcept;

  template <class T> [[nodiscard]] T *data() noexcept { return m_Storage->data<T>() + m_Offset; }
  template <class T> [[nodiscard]] const T *data() const noexcept { return m_Storage->data<const T>() + m_Offset; }

  template <class T, Index... Indices> T &at(Indices... indices);
  template <class T, Index... Indices> const T &at(Indices... indices) const;

  // Deep copy:
  template <class Self = TensorImpl> [[nodiscard]] memory::IntrusiveRef<Self> clone() const;

private:
  memory::IntrusiveRef<memory::Storage> m_Storage;
  Shape m_Shape;
  Strides m_Strides;
  std::size_t m_Offset{};
  DType m_Dtype;

  std::size_t ComputeStorageOffset(std::span<const std::size_t> indices) const;
  void CopyElementsInto(TensorImpl &dst) const;

  static void IncrementIndices(std::span<std::size_t> indices, const Shape &shape);
};
} // namespace ml::core

#include "TensorImpl.inl"
