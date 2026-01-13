#pragma once

#include <cstdint>
#include <vector>
#include <filesystem>

namespace logos {
template <typename value_type = double> class Matrix {
public:
  // Use RAII
  Matrix() = delete;
  explicit Matrix(uint32_t rows, uint32_t cols, const std::filesystem path);
  explicit Matrix(uint32_t rows, uint32_t cols);
  Matrix(Matrix &other);
  Matrix &operator=(Matrix &other);
  Matrix(Matrix &&other) = delete;
  Matrix &operator=(Matrix &&other) = delete;
  ~Matrix();

  Matrix operator+(Matrix &other);
  Matrix operator-(Matrix &other);
  Matrix operator*(Matrix &other);
  Matrix operator/(Matrix &other);

  void Clear();
  void Fill(value_type val);
  void Scale(value_type scale);
  void Transpose();

private:
  uint32_t m_Rows, m_Cols;
  std::vector<value_type> m_Data;
};
}; // namespace logos