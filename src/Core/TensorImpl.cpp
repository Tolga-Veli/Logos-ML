#include "Core/TensorImpl.hpp"
#include "Core/Error.hpp"

namespace ml::core {

TensorImpl::TensorImpl(const Shape &shape, DType dtype, memory::Device device)
    : m_Storage(memory::CreateIntrusiveRef<memory::Storage>(shape.num_elements() * dtype_size(dtype), device)),
      m_Shape(shape), m_Strides(Strides::Contiguous(shape)), m_Offset(0), m_Dtype(dtype) {}

TensorImpl::TensorImpl(memory::IntrusiveRef<memory::Storage> storage, const Shape &shape, const Strides &strides,
                       std::size_t offset, DType dtype) noexcept
    : m_Storage(std::move(storage)), m_Shape(shape), m_Strides(strides), m_Offset(offset), m_Dtype(dtype) {}

bool TensorImpl::is_contiguous() const noexcept {
  if (rank() == 0)
    return true;

  std::size_t exp = 1;
  for (std::size_t dim = rank(); dim-- > 0;) {
    if (m_Strides[dim] != exp)
      return false;

    exp *= m_Shape[dim];
  }
  return true;
}

std::size_t TensorImpl::ComputeStorageOffset(std::span<const std::size_t> indices) const {
  if (indices.size() != rank())
    throw ShapeError("TensorImpl: indices count must equal to the rank");

  std::size_t offset = 0;
  for (std::size_t i = 0; i < indices.size(); i++) {
    if (indices[i] >= m_Shape[i])
      throw std::out_of_range(
          std::format("TensorImpl: index {} out of bounds for dim {} with size {}", indices[i], i, m_Shape[i]));

    offset += indices[i] * m_Strides[i];
  }

  return offset;
}

// Walks *this via its real (possibly non-contiguous, non-unit-stride)
// multi-index and writes into dst's contiguous storage, row-major.
// dst must already be shaped/allocated to match *this.
void TensorImpl::CopyElementsInto(TensorImpl &dst) const {
  const std::size_t elemSize = dtype_size(m_Dtype);
  auto *src = m_Storage->raw_data(), *dstData = dst.m_Storage->raw_data();

  if (rank() == 0) {
    std::memcpy(dstData, src + m_Offset * elemSize, elemSize);
    return;
  }

  const auto lastDim = rank() - 1, innerExtent = m_Shape[lastDim], innerStride = m_Strides[lastDim],
             numRows = num_elements() / innerExtent;

  std::vector<std::size_t> indices(rank() - 1, 0);
  for (std::size_t row = 0; row < numRows; row++) {
    std::ptrdiff_t srcOffset = m_Offset;
    for (std::size_t dim = 0; dim < lastDim; dim++)
      srcOffset += indices[dim] * m_Strides[dim];

    std::byte *dstRow = dstData + row * innerExtent * elemSize;
    if (innerStride == 1) {
      std::memcpy(dstRow, src + srcOffset * elemSize, innerExtent * elemSize);
    } else {
      for (std::size_t i = 0; i < innerExtent; i++)
        std::memcpy(dstRow + i * elemSize, src + (srcOffset + i * innerStride) * elemSize, elemSize);
    }
    IncrementIndices(indices, m_Shape);
  }
}

void TensorImpl::IncrementIndices(std::span<std::size_t> indices, const Shape &shape) {
  for (std::size_t i = indices.size(); i-- > 0;) {
    indices[i]++;

    if (indices[i] < shape[i])
      return;

    indices[i] = 0;
  }
}
} // namespace ml::core
