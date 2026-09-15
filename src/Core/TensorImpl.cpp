#include "Core/TensorImpl.hpp"

namespace ml::core {

bool TensorImpl::is_contiguous() const noexcept {
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

std::size_t TensorImpl::ComputeStorageOffset(std::span<const int> indices) const {
  CORE_VERIFY(static_cast<int>(indices.size()) == rank(), "Indices count must equal to the rank");

  std::size_t offset{0};
  for (std::size_t i{0}; i < indices.size(); i++) {
    CORE_VERIFY(indices[i] >= 0 && indices[i] < m_Shape[i], "Trying to index out of the bounds of the tensor");
    offset += indices[i] * m_Strides[i];
  }
  return offset;
}

// Walks *this via its real (possibly non-contiguous, non-unit-stride)
// multi-index and writes into dst's contiguous storage, row-major.
// dst must already be shaped/allocated to match *this.
void TensorImpl::CopyElementsInto(TensorImpl &dst) const {
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

  const std::size_t lastDim = rank() - 1, innerExtent = m_Shape[lastDim], innerStride = m_Strides[lastDim],
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
        std::memcpy(dstRow + i * elemSize, src + (srcOffset + i * innerStride) * elemSize, elemSize);
    }
    IncrementIndices(indices, m_Shape);
  }
}

void TensorImpl::IncrementIndices(std::span<int> indices, const Shape &shape) {
  for (int i = static_cast<int>(indices.size()) - 1; i >= 0; i--) {
    indices[i]++;
    if (indices[i] < shape[i])
      return;

    indices[i] = 0;
  }
}
} // namespace ml::core
