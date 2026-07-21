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
  Tensor()
      : m_Storage(std::make_shared<Storage<T>>(0)), m_Shape(), m_Strides(),
        m_Offset() {}

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

  [[nodiscard]] size_t rank() const { return m_Shape.rank(); }
  [[nodiscard]] size_t num_elements() const { return m_Shape.num_elements(); }

  [[nodiscard]] T *data() { return m_Storage->data() + m_Offset; }
  [[nodiscard]] const T *data() const { return m_Storage->data() + m_Offset; }

  [[nodiscard]] size_t offset() const { return m_Offset; }

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

    for (auto &x : *this)
      x = value;
  }

  // Checks whether memory layout matches a standard row-major layout.
  [[nodiscard]]
  bool IsContiguous() const {
    if (rank() == 0)
      return true;

    size_t expected = 1;
    for (size_t idx = rank(); idx-- > 0;) {
      if (m_Strides[idx] != expected)
        return false;
      expected *= m_Shape[idx];
    }
    return true;
  }

  /* Transpose does NOT move or copy data.
     It creates a new Tensor with new metadata around the same memory
     and changes the way we interpret the Tensor

     Example:
     Shape:    [2,3]
     Strides:  [3,1]

     After swapping dim0 and dim1:
     Shape:    [3,2]
     Strides:  [1,3]
  */
  [[nodiscard]] Tensor Transpose(size_t dim0, size_t dim1) const {
    assert(dim0 < rank() && dim1 < rank());

    auto shape = m_Shape.dims();
    auto strides = m_Strides.values();

    std::swap(shape[dim0], shape[dim1]);
    std::swap(strides[dim0], strides[dim1]);

    return Tensor(m_Storage, Shape(std::move(shape)),
                  Strides(std::move(strides)), m_Offset);
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
  [[nodiscard]] Tensor Slice(size_t dim, size_t start, size_t end) const {
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
    requires(std::convertible_to<Indices, std::size_t> && ...)
  T &operator()(Indices... indices) {
    std::array<std::size_t, sizeof...(Indices)> idx{
        static_cast<std::size_t>(indices)...};
    return data()[ComputeOffset(idx) - m_Offset];
  }

  template <typename... Indices>
    requires(std::convertible_to<Indices, std::size_t> && ...)
  const T &operator()(Indices... indices) const {
    std::array<std::size_t, sizeof...(Indices)> idx{
        static_cast<std::size_t>(indices)...};

    return data()[ComputeOffset(idx) - m_Offset];
  }

  // Forward-declared so Tensor::begin()/end() can name it; defined below
  // once the class's private helpers it relies on are visible.
  template <class type> class Iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = type;
    using reference = type &;
    using difference_type = std::ptrdiff_t;
    using pointer = type *;

    Iterator(Tensor *_tensor, bool isEnd)
        : tensor(_tensor), index(tensor->rank(), 0),
          done(isEnd || tensor->num_elements() == 0) {}

    reference operator*() { return tensor->data()[ComputeOffset()]; }

    Iterator &operator++() {
      if (tensor->rank() == 0) {
        done = true;
        return *this;
      }

      for (size_t i = index.size(); i-- > 0;) {
        index[i]++;
        if (index[i] < tensor->GetShape()[i])
          return *this;
        index[i] = 0;
      }

      // Carried out of the most-significant dimension: we've visited
      // every element.
      done = true;
      return *this;
    }

    [[nodiscard]] bool operator!=(const Iterator &other) const {
      return tensor == other.tensor && index == other.index &&
             done == other.done;
    }

    [[nodiscard]] bool operator==(const Iterator &other) const {
      return done == other.done;
    }

  private:
    Tensor *tensor;
    // HACK: to reduce heap allocations i have made this into an array
    // but it might go out of bounds since i dont know how large i should make
    // the array
    std::array<size_t, 512> index;
    bool done;

    [[nodiscard]] size_t ComputeOffset() const {
      size_t offset = tensor->offset();
      const auto &strides = tensor->GetStrides();
      for (size_t i = 0; i < index.size(); i++)
        offset += index[i] * strides[i];
      return offset;
    }
  };

  using iterator = Iterator<T>;
  using const_iterator = Iterator<const T>;

  iterator begin() { return iterator(this, false); }
  iterator end() { return iterator(this, true); }

  const_iterator begin() const { return const_iterator(this, false); }
  const_iterator end() const { return const_iterator(this, true); }

private:
  std::shared_ptr<Storage<T>> m_Storage;
  Shape m_Shape;
  Strides m_Strides;
  std::size_t m_Offset = 0;

  Tensor(std::shared_ptr<Storage<T>> storage, const Shape &shape,
         const Strides &strides, std::size_t offset)
      : m_Storage(std::move(storage)), m_Shape(shape), m_Strides(strides),
        m_Offset(offset) {}

  [[nodiscard]] size_t ComputeOffset(std::span<const size_t> indices) const {
    assert(indices.size() == rank());

    size_t offset = m_Offset;
    for (size_t i = 0; i < indices.size(); i++) {
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
    size_t n = other.num_elements();
    if (n == 0)
      return;

    if (other.rank() == 0) {
      data()[0] = other.data()[0];
      return;
    }

    std::vector<size_t> idx(other.rank(), 0);
    for (size_t flat = 0; flat < n; flat++) {
      size_t srcOff = other.m_Offset;
      for (size_t d = 0; d < other.rank(); d++)
        srcOff += idx[d] * other.m_Strides[d];

      data()[flat] = other.m_Storage->data()[srcOff];

      for (size_t d = other.rank(); d-- > 0;) {
        idx[d]++;
        if (idx[d] < other.m_Shape[d])
          break;
        idx[d] = 0;
      }
    }
  }
};
} // namespace ml::core
