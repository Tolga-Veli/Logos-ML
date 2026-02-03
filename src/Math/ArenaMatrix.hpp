#pragma once
#include "Memory/Arena.hpp" // your Logos::Memory::Arena
#include "MatrixView.hpp"
#include <cstddef>

namespace Logos {

template <class T> class ArenaMatrix {
public:
  ArenaMatrix() = delete;
  ArenaMatrix(Logos::Memory::Arena &arena, std::size_t rows, std::size_t cols)
      : m_Arena(&arena), m_Rows(rows), m_Cols(cols) {
    m_Data = arena.allocate<T>(rows * cols, alignof(T));
  }

  std::size_t rows() const noexcept { return m_Rows; }
  std::size_t cols() const noexcept { return m_Cols; }
  T *data() noexcept { return m_Data; }
  const T *data() const noexcept { return m_Data; }

  MatrixView<T> view() noexcept {
    return MatrixView<T>::row_major(m_Data, m_Rows, m_Cols);
  }

  MatrixView<const T> view() const noexcept {
    return MatrixView<const T>::row_major(m_Data, m_Rows, m_Cols);
  }

private:
  Logos::Memory::Arena *m_Arena;
  std::size_t m_Rows, m_Cols;
  T *m_Data;
};
} // namespace Logos
