#pragma once

#include "Core/Assert.hpp"
#include "Core/DType.hpp"
#include "Core/Shape.hpp"
#include "Core/Tensor.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <numeric>
#include <random>
#include <vector>

namespace ml::core {

template <class T = float>
inline Tensor load_binary(const std::filesystem::path &path, std::initializer_list<int> shape) {
  Tensor tensor{core::Shape(shape), core::dtype_of<T>()};

  std::ifstream file(path, std::ios::binary);
  CORE_VERIFY(file, "Cannot open: " + path.string());

  const auto nbytes = sizeof(T) * tensor.num_elements();
  file.read(reinterpret_cast<char *>(tensor.data<T>()), nbytes);
  CORE_VERIFY(file, "Read failed: " + path.string());
  return tensor;
}

// A single batch of image data
struct Batch {
  Tensor images, labels;
};

class DataLoader {
public:
  DataLoader(Tensor images, Tensor labels, int batch_size, bool shuffle = true);

  int num_batches() const { return m_Count / m_BatchSize; }
  int count() const noexcept { return m_Count; }
  int batch_size() const noexcept { return m_BatchSize; }

  // Returns false when epoch is done
  bool next(Batch &out);
  void reset();

private:
  Tensor m_Images, m_Labels;
  int m_BatchSize, m_Features = 0, m_Count = 0, m_Cursor = 0;
  bool m_Shuffle;

  std::vector<int> m_Indices;
  std::mt19937 m_Rng;
};
} // namespace ml::core
