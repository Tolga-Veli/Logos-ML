#pragma once

#include "Core/Shape.hpp"
#include "Core/Tensor.hpp"

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <vector>

namespace ml::data {
using core::Tensor;

template <class T = float>
inline Tensor<T> load_binary(const std::filesystem::path &path,
                             std::initializer_list<int> shape) {
  Tensor<T> tensor{core::Shape(shape)};

  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open: " + path.string());

  const auto nbytes = sizeof(T) * tensor.num_elements();
  f.read(reinterpret_cast<char *>(tensor.data()), nbytes);

  if (!f)
    throw std::runtime_error("Read failed: " + path.string());

  return tensor;
}

// A single batch of image data
template <class T = float> struct Batch {
  Tensor<T> images;
  Tensor<int> labels;
};

template <class T = float> class DataLoader {
public:
  DataLoader(Tensor<T> images, Tensor<int> labels, int batch_size,
             bool shuffle = true)
      : m_Images(std::move(images)), m_Labels(std::move(labels)),
        m_BatchSize(batch_size), m_Features(m_Images.shape()[1]),
        m_Count(m_Images.shape()[0]), m_Shuffle(shuffle), m_Indices(m_Count),
        m_Rng(std::random_device{}()) {
    assert(m_Labels.shape()[0] == m_Count);
    std::iota(m_Indices.begin(), m_Indices.end(), 0);
  }

  // Total number of complete batches
  int num_batches() const { return m_Count / m_BatchSize; }
  int count() const { return m_Count; }
  int batch_size() const { return m_BatchSize; }

  // Shuffles indices — call at the start of each epoch
  void reset() {
    if (m_Shuffle)
      std::shuffle(m_Indices.begin(), m_Indices.end(), m_Rng);
    m_Cursor = 0;
  }

  // Returns false when epoch is done
  bool next(Batch<T> &out) {
    if (m_Cursor + m_BatchSize > m_Count)
      return false;

    Tensor<T> batch_images({m_BatchSize, m_Features});
    Tensor<int> batch_labels({m_BatchSize});

    for (int i = 0; i < m_BatchSize; ++i) {
      const int idx = m_Indices[m_Cursor + i];

      std::copy_n(m_Images.data() + idx * m_Features, m_Features,
                  batch_images.data() + i * m_Features);

      batch_labels.data()[i] = m_Labels.data()[idx];
    }

    out = {std::move(batch_images), std::move(batch_labels)};
    m_Cursor += m_BatchSize;
    return true;
  }

private:
  Tensor<T> m_Images;
  Tensor<int> m_Labels;
  int m_BatchSize, m_Features, m_Count, m_Cursor = 0;
  bool m_Shuffle;

  std::vector<int> m_Indices;
  std::mt19937 m_Rng;
};
} // namespace ml::data
