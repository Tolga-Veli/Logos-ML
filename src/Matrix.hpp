#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>

namespace linalg {
class Matrix {
public:
  Matrix();
  Matrix(uint64_t rows, uint64_t cols);

  uint64_t GetRows() const { return m_Rows; }
  uint64_t GetCols() const { return m_Cols; }
  size_t GetSize() const { return m_Data.size(); }

  float &at(uint64_t row, uint64_t col);
  const float &at(uint64_t row, uint64_t col) const;

  void Fill(float val);

  float *data() { return m_Data.data(); }
  const float *data() const { return m_Data.data(); }

private:
  std::vector<float> m_Data;
  uint64_t m_Rows, m_Cols;
};
} // namespace linalg