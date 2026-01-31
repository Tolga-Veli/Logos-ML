#pragma once

#include <cassert>
#include <cstddef>

#include "Buffer.hpp"
#include "MatrixView.hpp"

namespace Logos::linalg {
// Matrix has a row-major memory layout
template <class T> class Matrix {
public:
  Matrix() = default;
  explicit Matrix(std::size_t rows, std::size_t cols,
                  std::size_t alignment = Logos::Memory::DEFAULT_ALIGNMENT);

  Matrix(const Matrix &other) = delete;
  Matrix &operator=(const Matrix &other) = delete;

  Matrix(Matrix &&other) noexcept;
  Matrix &operator=(Matrix &&other) noexcept;

  T &operator()(std::size_t row, std::size_t col) {
    return m_Buffer.as<T>()[row * m_LeadingDim + col];
  }
  const T &operator()(std::size_t row, std::size_t col) const {
    return m_Buffer.as<T>()[row * m_LeadingDim + col];
  }

  std::size_t rows() const noexcept { return m_Rows; }
  std::size_t cols() const noexcept { return m_Cols; }
  std::size_t size() const noexcept { return m_Rows * m_Cols; }

  std::size_t size_bytes() const noexcept { return m_Buffer.size(); }
  std::size_t leading_dim() const noexcept { return m_LeadingDim; }

  T *data() noexcept { return m_Buffer.as<T>(); }
  const T *data() const noexcept { return m_Buffer.as<T>(); }

  MatrixView<T> view() {
    return MatrixView<T>(m_Buffer.as<T>(), m_Rows, m_Cols, m_LeadingDim, 1);
  }

  MatrixView<const T> view() const {
    return MatrixView<const T>(m_Buffer.as<T>(), m_Rows, m_Cols, m_LeadingDim,
                               1);
  }

  void fill_zeroes() { m_Buffer.clear(); }

private:
  Logos::Memory::Buffer m_Buffer;
  std::size_t m_Rows = 0, m_Cols = 0, m_LeadingDim = 0;
};
} // namespace Logos::linalg
