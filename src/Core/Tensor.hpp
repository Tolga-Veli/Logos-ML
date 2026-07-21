#pragma once

#include "Shape.hpp"
#include "Storage.hpp"
#include "Strides.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <span>
#include <vector>

namespace ml::core {

template <class T = float> class Tensor {
public:
  Tensor() : Tensor(Shape{}) {}

  Tensor(const Shape &shape)
      : m_Storage(std::make_shared<Storage<T>>(shape.num_elements())),
        m_Shape(shape), m_Strides(Strides::Contiguous(shape)), m_Offset(0) {}

  Tensor(const Shape &shape, const T &value) : Tensor(shape) { Fill(value); }

  Tensor(const Tensor &other)
      : m_Storage(std::make_shared<Storage<T>>(other.num_elements())),
        m_Shape(other.m_Shape), m_Strides(Strides::Contiguous(other.m_Shape)),
        m_Offset(0) {
    CopyElementsFrom(other);
  }

  Tensor &operator=(const Tensor &other) {
    if (this == &other)
      return *this;

    m_Shape = other.m_Shape;
    m_Strides = Strides::Contiguous(other.m_Shape);
    m_Offset = 0;

    m_Storage = std::make_shared<Storage<T>>(other.num_elements());

    CopyElementsFrom(other);

    return *this;
  }

  Tensor(Tensor &&other) noexcept
      : m_Storage(std::move(other.m_Storage)),
        m_Shape(std::move(other.m_Shape)),
        m_Strides(std::move(other.m_Strides)), m_Offset(other.m_Offset) {
    other.m_Offset = 0;
  }

  Tensor &operator=(Tensor &&other) noexcept {
    if (this == &other)
      return *this;

    m_Storage = std::move(other.m_Storage);
    m_Shape = std::move(other.m_Shape);
    m_Strides = std::move(other.m_Strides);
    m_Offset = other.m_Offset;

    other.m_Offset = 0;

    return *this;
  }

  ~Tensor() = default;

  [[nodiscard]] int rank() const { return m_Shape.rank(); }
  [[nodiscard]] int num_elements() const { return m_Shape.num_elements(); }

  [[nodiscard]] T *data() { return m_Storage->data() + m_Offset; }
  [[nodiscard]] const T *data() const { return m_Storage->data() + m_Offset; }

  [[nodiscard]] int offset() const { return m_Offset; }

  [[nodiscard]]
  const Shape &GetShape() const {
    return m_Shape;
  }

  [[nodiscard]]
  const Strides &GetStrides() const {
    return m_Strides;
  }

  [[nodiscard]]
  const Storage<T> &GetStorage() const {
    return *m_Storage;
  }

  void Fill(const T &value) {
    if (IsContiguous()) {
      std::fill_n(data(), num_elements(), value);
      return;
    }

    if (rank() == 0) {
      m_Storage->data()[m_Offset] = value;
      return;
    }

    const int innerExtent = m_Shape[rank() - 1];
    const int numRows = num_elements() / innerExtent;

    std::vector<int> indices(rank() - 1, 0);
    for (int row{0}; row < numRows; row++) {
      int offset = m_Offset;
      for (int dim{0}; dim < rank() - 1; dim++)
        offset += indices[dim] * m_Strides[dim];

      std::fill_n(m_Storage->data() + offset, innerExtent, value);
      IncrementIndices(indices, m_Shape);
    }
  }

  // Checks whether memory layout matches a standard row-major layout.
  [[nodiscard]]
  bool IsContiguous() const {
    if (rank() == 0)
      return true;

    int expected = 1;
    for (int idx = rank() - 1; idx > 0; idx--) {
      if (m_Strides[idx] != expected)
        return false;

      expected *= m_Shape[idx];
    }

    return true;
  }

  /* Slice does NOT move or copy data.
     It creates a new Tensor with new metadata around the same memory and
     changes the way we interpret the Tensor

     It works by:
     1. Reducing the size of one dimension
     2. Moving the starting offset forward

     Example:
     Shape:   [10,5]
     Slice(0,2,6)

     New shape becomes:
     [4,5]

     And offset shifts:
     offset += start * stride[dim]
  */
  [[nodiscard]] Tensor Slice(int dim, int start, int end) const {
    assert(dim < rank());
    assert(start <= end);
    assert(end <= m_Shape[dim]);

    auto shape = m_Shape.dims();
    shape[dim] = end - start;

    return Tensor(m_Storage, Shape(std::move(shape)), m_Strides,
                  m_Offset + start * m_Strides[dim]);
  }

  /*  Reshape does NOT move or copy data.
      It creates a new Tensor with new metadata around the same memory and
      changes the way we interpret the Tensor

      IMPORTANT:
      This only works if the tensor is contiguous in memory.

      Example:
      Shape:    [2,3]
      Data:     [1, 2, 3, 4, 5, 6]

      Reshape to [3,2] becomes:
      [1, 2]
      [3, 4]
      [5, 6]
  */
  [[nodiscard]]
  Tensor Reshape(const Shape &newShape) const {
    assert(IsContiguous());
    assert(newShape.num_elements() == num_elements());

    return Tensor(m_Storage, newShape, Strides::Contiguous(newShape), m_Offset);
  }

  // T(1,2,3,...) access
  template <typename... Indices>
    requires(std::convertible_to<Indices, int> && ...)
  T &operator()(Indices... indices) {
    std::array<int, sizeof...(Indices)> idx{static_cast<int>(indices)...};
    return data()[ComputeOffset(idx)];
  }

  template <typename... Indices>
    requires(std::convertible_to<Indices, int> && ...)
  const T &operator()(Indices... indices) const {
    std::array<int, sizeof...(Indices)> idx{static_cast<int>(indices)...};

    return data()[ComputeOffset(idx)];
  }

private:
  std::shared_ptr<Storage<T>> m_Storage;
  Shape m_Shape;
  Strides m_Strides;
  int m_Offset = 0;

  Tensor(std::shared_ptr<Storage<T>> storage, const Shape &shape,
         const Strides &strides, int offset)
      : m_Storage(std::move(storage)), m_Shape(shape), m_Strides(strides),
        m_Offset(offset) {}

  [[nodiscard]] int ComputeOffset(std::span<const int> indices) const {
    assert(indices.size() == rank());

    int offset{0};
    for (int i = 0; i < indices.size(); i++) {
      assert(indices[i] < m_Shape[i]);
      offset += indices[i] * m_Strides[i];
    }

    return offset;
  }

  // Walks `other` via its real (possibly non-contiguous) multi-index and
  // writes each element into *this* contiguous storage in row-major
  // order. Used by the copy ctor / copy-assignment fix above. Assumes
  // *this has already been sized to other.num_elements() and uses
  // Strides::Contiguous layout.
  void CopyElementsFrom(const Tensor &other) {
    int n = other.num_elements();
    if (n == 0)
      return;

    if (other.rank() == 0) {
      data()[0] = other.data()[0];
      return;
    }

    const int innerExtent = other.m_Shape[other.rank() - 1];
    const int numRows = n / innerExtent;

    std::vector<int> indices(other.rank() - 1, 0);
    T *dst = data(); // *this is freshly allocated & contiguous, per the
                     // ctor/assign invariant
    for (int row{0}; row < numRows; row++) {
      int srcOffset{0};
      for (int dim{0}; dim < other.rank() - 1; dim++)
        srcOffset += indices[dim] * other.m_Strides[dim];

      std::copy_n(other.data() + srcOffset, innerExtent,
                  dst + row * innerExtent);

      IncrementIndices(indices, other.m_Shape);
    }
  }

  void IncrementIndices(std::span<int> indices, const Shape &shape) {
    for (int i = indices.size(); i-- > 0;) {
      indices[i]++;
      if (indices[i] < shape[i])
        return;

      indices[i] = 0;
    }
  }
};
} // namespace ml::core
