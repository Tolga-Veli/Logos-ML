#pragma once

#include "Core/Assert.hpp"
#include "Core/DType.hpp"
#include "Core/Shape.hpp"
#include "Core/Strides.hpp"
#include "Memory/Device.hpp"
#include "Memory/IntrusiveRef.hpp"
#include "Memory/Storage.hpp"

#include <algorithm>
#include <cstring>
#include <span>
#include <vector>

namespace ml::core {
/*
  TensorImpl owns the actual tensor data: storage, shape, strides, offset.
  It is never held by value - always accessed through
  IntrusiveRef<TensorImpl<T>>. Tensor<T> is the thin, cheaply-copyable handle
  around it (see Tensor.hpp)

  Non-copyable, non-movable
*/
class TensorImpl : public memory::detail::IntrusiveRefCounted {
public:
  explicit TensorImpl(const Shape &shape, DType dtype)
      : m_Storage(memory::CreateIntrusiveRef<memory::Storage>(shape.num_elements() * dtype_size(dtype))),
        m_Shape(shape), m_Strides(Strides::Contiguous(shape)), m_Offset(0), m_Dtype(dtype) {}

  TensorImpl(memory::IntrusiveRef<memory::Storage> storage, const Shape &shape, const Strides &strides,
             std::size_t offset, DType dtype) noexcept
      : m_Storage(std::move(storage)), m_Shape(shape), m_Strides(strides), m_Offset(offset), m_Dtype(dtype) {}

  TensorImpl(const TensorImpl &) = delete;
  TensorImpl &operator=(const TensorImpl &) = delete;
  TensorImpl(TensorImpl &&) = delete;
  TensorImpl &operator=(TensorImpl &&) = delete;

  [[nodiscard]] int rank() const noexcept { return m_Shape.rank(); }
  [[nodiscard]] int num_elements() const noexcept { return m_Shape.num_elements(); }
  [[nodiscard]] std::size_t offset() const noexcept { return m_Offset; }
  [[nodiscard]] const Shape &shape() const noexcept { return m_Shape; }
  [[nodiscard]] const Strides &strides() const noexcept { return m_Strides; }
  [[nodiscard]] memory::Device device() const noexcept { return m_Storage->device(); }
  [[nodiscard]] DType dtype() const noexcept { return m_Dtype; }
  [[nodiscard]] std::byte *raw_data() noexcept { return m_Storage->data() + m_Offset * dtype_size(m_Dtype); }
  [[nodiscard]] const std::byte *raw_data() const noexcept {
    return m_Storage->data() + m_Offset * dtype_size(m_Dtype);
  }
  [[nodiscard]] bool is_contiguous() const noexcept;

  template <class T> [[nodiscard]] T *data() noexcept { return m_Storage->as<T>() + m_Offset; }
  template <class T> [[nodiscard]] const T *data() const noexcept { return m_Storage->as<const T>() + m_Offset; }

  template <class T, typename... Indices>
    requires(std::convertible_to<Indices, int> && ...)
  T &operator()(Indices... indices) {
    std::array<int, sizeof...(Indices)> idx{static_cast<int>(indices)...};
    return data<T>()[ComputeStorageOffset(idx)];
  }

  template <class T, typename... Indices>
    requires(std::convertible_to<Indices, int> && ...)
  const T &operator()(Indices... indices) const {
    std::array<int, sizeof...(Indices)> idx{static_cast<int>(indices)...};
    return data<T>()[ComputeStorageOffset(idx)];
  }

  // Deep copy:
  template <class Self = TensorImpl> [[nodiscard]] memory::IntrusiveRef<Self> clone() const {
    auto out = ml::memory::CreateIntrusiveRef<Self>(m_Shape, m_Dtype);
    CopyElementsInto(*out);
    return out;
  }

private:
  memory::IntrusiveRef<memory::Storage> m_Storage;
  Shape m_Shape;
  Strides m_Strides;
  std::size_t m_Offset;
  DType m_Dtype;

  std::size_t ComputeStorageOffset(std::span<const int> indices) const;
  void CopyElementsInto(TensorImpl &dst) const;

  static void IncrementIndices(std::span<int> indices, const Shape &shape);
};
} // namespace ml::core
