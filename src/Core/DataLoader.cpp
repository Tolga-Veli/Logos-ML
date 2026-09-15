#include "Core/DataLoader.hpp"

namespace ml::core {

DataLoader::DataLoader(Tensor images, Tensor labels, int batch_size, bool shuffle = true)
    : m_Images(std::move(images)), m_Labels(std::move(labels)), m_BatchSize(batch_size), m_Shuffle(shuffle),
      m_Rng(std::random_device{}()) {

  CORE_VERIFY(m_Images.rank() == 2 && m_Labels.rank() == 1, "DataLoader expects images [N, features] and labels [N]");
  CORE_VERIFY(m_BatchSize > 0, "batch_size must be positive");

  m_Count = m_Images.shape()[0];
  m_Features = m_Images.shape()[1];

  CORE_VERIFY(m_Labels.shape()[0] == m_Count, "images/labels count mismatch");
  CORE_VERIFY(m_Images.is_contiguous() && m_Labels.is_contiguous(), "DataLoader requires contiguous input tensors");

  m_Indices.resize(m_Count);
  std::iota(m_Indices.begin(), m_Indices.end(), 0);
}

int num_batches() const { return m_Count / m_BatchSize; }
int count() const noexcept { return m_Count; }
int batch_size() const noexcept { return m_BatchSize; }

void DataLoader::reset() {
  if (m_Shuffle)
    std::shuffle(m_Indices.begin(), m_Indices.end(), m_Rng);
  m_Cursor = 0;
}

// Returns false when epoch is done
bool DataLoader::next(Batch &out) {
  if (m_Cursor + m_BatchSize > m_Count)
    return false;

  const auto imgType = m_Images.dtype(), labelType = m_Labels.dtype();
  const std::size_t imgElemSize = core::dtype_size(imgType), labelElemSize = core::dtype_size(labelType);

  Tensor batch_images({m_BatchSize, m_Features}, imgType), batch_labels({m_BatchSize}, labelType);

  std::byte *dstImg = batch_images.raw_data(), *dstLbl = batch_labels.raw_data();
  const std::byte *srcImg = m_Images.raw_data(), *srcLbl = m_Labels.raw_data();

  for (int i = 0; i < m_BatchSize; ++i) {
    const int idx = m_Indices[m_Cursor + i];
    std::memcpy(dstImg + static_cast<std::size_t>(i) * m_Features * imgElemSize,
                srcImg + static_cast<std::size_t>(idx) * m_Features * imgElemSize, m_Features * imgElemSize);
    std::memcpy(dstLbl + static_cast<std::size_t>(i) * labelElemSize,
                srcLbl + static_cast<std::size_t>(idx) * labelElemSize, labelElemSize);
  }

  out = {std::move(batch_images), std::move(batch_labels)};
  m_Cursor += m_BatchSize;
  return true;
}
} // namespace ml::core
