#pragma once

#include "Core/Assert.hpp"
#include "DType.hpp"
#include "Memory/Device.hpp"
#include "Memory/IntrusiveRef.hpp"
#include "Memory/Storage.hpp"
#include "Shape.hpp"
#include "Strides.hpp"

#include <algorithm>
#include <cassert>
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
      : m_Storage(memory::CreateIntrusiveRef<memory::Storage>(
            shape.num_elements() * dtype_size(dtype))),
        m_Shape(shape), m_Strides(Strides::Contiguous(shape)), m_Offset(0),
        m_Dtype(dtype) {}

  TensorImpl(memory::IntrusiveRef<memory::Storage> storage, const Shape &shape,
             const Strides &strides, std::size_t offset, DType dtype) noexcept
      : m_Storage(std::move(storage)), m_Shape(shape), m_Strides(strides),
        m_Offset(offset), m_Dtype(dtype) {}

  TensorImpl(const TensorImpl &) = delete;
  TensorImpl &operator=(const TensorImpl &) = delete;
  TensorImpl(TensorImpl &&) = delete;
  TensorImpl &operator=(TensorImpl &&) = delete;

  [[nodiscard]] int rank() const noexcept { return m_Shape.rank(); }
  [[nodiscard]] int num_elements() const noexcept {
    return m_Shape.num_elements();
  }

  template <class T> [[nodiscard]] T *data() noexcept {
    CORE_ASSERT(dtype() == dtype_of<T>(),
                "Requested data type does not match tensor dtype");
    return m_Storage->as<T>() + m_Offset;
  }

  template <class T> [[nodiscard]] const T *data() const noexcept {
    CORE_ASSERT(dtype() == dtype_of<T>(),
                "Requested data type does not match tensor dtype");
    return m_Storage->as<const T>() + m_Offset;
  }

  std::byte *raw_data() noexcept { return m_Storage->data(); }
  const std::byte *raw_data() const noexcept { return m_Storage->data(); }

  [[nodiscard]] std::size_t offset() const noexcept { return m_Offset; }
  [[nodiscard]] const Shape &shape() const noexcept { return m_Shape; }
  [[nodiscard]] const Strides &strides() const noexcept { return m_Strides; }

  [[nodiscard]] memory::Device device() const noexcept {
    return m_Storage->device();
  }

  [[nodiscard]] DType dtype() const noexcept { return m_Dtype; }

  template <class T> void fill(T value) {
    CORE_ASSERT(dtype() == m_Dtype, "fill value dtype must match tensor dtype");
    T *ptr = data<T>();
    if (is_contiguous()) {
      std::fill_n(ptr, num_elements(), value);
      return;
    }

    if (rank() == 0) {
      ptr[m_Offset] = value;
      return;
    }

    const auto lastDim = rank() - 1, innerExtent = m_Shape[lastDim],
               innerStride = m_Strides[lastDim],
               numRows = num_elements() / innerExtent;

    std::vector<int> indices(rank() - 1, 0);
    for (int row{0}; row < numRows; row++) {
      int offset = m_Offset;
      for (int dim{0}; dim < lastDim; dim++)
        offset += indices[dim] * m_Strides[dim];

      if (innerStride == 1)
        std::fill_n(ptr + offset, innerExtent, value);
      else
        for (int i{0}; i < innerExtent; i++)
          ptr[offset + i * innerStride] = value;

      IncrementIndices(indices, m_Shape);
    }
  }

  [[nodiscard]] bool is_contiguous() const noexcept {
    if (rank() == 0)
      return true;

    int expected = 1;
    for (int dim = rank() - 1; dim >= 0; dim--) {
      if (m_Strides[dim] != expected)
        return false;
      expected *= m_Shape[dim];
    }
    return true;
  }

  template <DType type, typename... Indices>
    requires(std::convertible_to<Indices, int> && ...)
  dtype_to_cpp_v<type> &operator()(Indices... indices) {
    std::array<int, sizeof...(Indices)> idx{static_cast<int>(indices)...};
    return data<type>()[ComputeStorageOffset(idx)];
  }

  template <DType type, typename... Indices>
    requires(std::convertible_to<Indices, int> && ...)
  const dtype_to_cpp_v<type> &operator()(Indices... indices) const {
    std::array<int, sizeof...(Indices)> idx{static_cast<int>(indices)...};
    return data<type>()[ComputeStorageOffset(idx)];
  }

  // Deep copy: new contiguous storage, elements copied out of *this
  // (possibly non-contiguous) view.
  // Templated on `Self` (defaulted to TensorImpl) rather than naming
  // TensorImpl directly in the return type. Naming IntrusiveRef<TensorImpl>
  // here would force checking IntrusiveRef's concept against TensorImpl
  // while TensorImpl is still incomplete (mid-definition)-circularity. Making
  // the return type depend on an uninstantiated template parameter defers the
  // check to first call, by which point TensorImpl is complete.
  template <class Self = TensorImpl>
  [[nodiscard]] memory::IntrusiveRef<Self> clone() const {
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

  std::size_t ComputeStorageOffset(std::span<const int> indices) const {
    CORE_ASSERT(static_cast<int>(indices.size()) == rank(),
                "Indices count must equal to the rank");

    std::size_t offset{0};
    for (std::size_t i{0}; i < indices.size(); i++) {
      CORE_ASSERT(indices[i] < m_Shape[i],
                  "Trying to index out of the bounds of the tensor");
      offset += indices[i] * m_Strides[i];
    }
    return offset;
  }

  // Walks *this via its real (possibly non-contiguous, non-unit-stride)
  // multi-index and writes into dst's contiguous storage, row-major.
  // dst must already be shaped/allocated to match *this.
  void CopyElementsInto(TensorImpl &dst) const {
    const auto n = num_elements();
    if (n == 0)
      return;

    const std::size_t elemSize = dtype_size(m_Dtype);
    auto *src = m_Storage->data();
    auto *dstData = dst.m_Storage->data();

    if (rank() == 0) {
      std::memcpy(dstData, src + m_Offset * elemSize, elemSize);
      return;
    }

    const std::size_t lastDim = rank() - 1, innerExtent = m_Shape[lastDim],
                      innerStride = m_Strides[lastDim],
                      numRows = n / innerExtent;

    std::vector<int> indices(rank() - 1, 0);
    for (std::size_t row{0}; row < numRows; row++) {
      std::size_t srcOffset = m_Offset;
      for (std::size_t dim{0}; dim < lastDim; dim++)
        srcOffset += indices[dim] * m_Strides[dim];

      std::byte *dstRow = dstData + row * innerExtent * elemSize;
      if (innerStride == 1) {
        std::memcpy(dstRow, src + srcOffset * elemSize, innerExtent * elemSize);
      } else {
        for (std::size_t i{0}; i < innerExtent; i++)
          std::memcpy(dstRow + i * elemSize,
                      src + (srcOffset + i * innerStride) * elemSize, elemSize);
      }
      IncrementIndices(indices, m_Shape);
    }
  }

  static void IncrementIndices(std::span<int> indices, const Shape &shape) {
    for (int i = static_cast<int>(indices.size()) - 1; i >= 0; i--) {
      indices[i]++;
      if (indices[i] < shape[i])
        return;

      indices[i] = 0;
    }
  }
};
} // namespace ml::core
