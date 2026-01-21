#include "Matrix.hpp"

#include <utility>

namespace Logos::linalg {
template <class T>
Matrix<T>::Matrix(std::size_t rows, std::size_t cols, std::size_t alignment)
    : m_Buffer(sizeof(T) * rows * cols, alignment), m_Rows(rows), m_Cols(cols),
      m_LeadingDim(cols) {}

template <class T>
Matrix<T>::Matrix(Matrix &&other) noexcept
    : m_Buffer(std::move(other.m_Buffer)), m_Rows(other.m_Rows),
      m_Cols(other.m_Cols), m_LeadingDim(other.m_LeadingDim) {
  other.m_Rows = other.m_Cols = other.m_LeadingDim = 0;
}

template <class T> Matrix<T> &Matrix<T>::operator=(Matrix &&other) noexcept {
  if (this == &other)
    return *this;

  m_Buffer = std::move(other.m_Buffer);
  m_Rows = other.m_Rows;
  m_Cols = other.m_Cols;
  m_LeadingDim = other.m_LeadingDim;

  other.m_Rows = other.m_Cols = other.m_LeadingDim = 0;
  return *this;
}
} // namespace Logos::linalg
