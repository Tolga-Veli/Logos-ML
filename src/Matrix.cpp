#include "Matrix.hpp"

namespace linalg {
Matrix::Matrix() : m_Rows(), m_Cols(), m_Data() {}
Matrix::Matrix(uint64_t rows, uint64_t cols)
    : m_Rows(rows), m_Cols(cols), m_Data(rows * cols) {}

float &Matrix::at(uint64_t row, uint64_t col) {
  return m_Data[row * m_Cols + col];
}
const float &Matrix::at(uint64_t row, uint64_t col) const {
  return m_Data[row * m_Cols + col];
}

void Matrix::Fill(float val) { std::fill(m_Data.begin(), m_Data.end(), val); }

} // namespace linalg