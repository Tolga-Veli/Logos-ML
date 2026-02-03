#pragma once

#include <cstddef>
#include <stdexcept>

#include "MatrixView.hpp"
#include "Memory/Arena.hpp"
#include "Memory/ScratchArena.hpp"

namespace Logos::linalg {
/* Non-owning matrix that lives on the main Arena*/
template <class T> class ArenaMatrix {
public:
  ArenaMatrix() = default;

  explicit ArenaMatrix(Memory::Arena &arena) noexcept : m_Arena(&arena) {}

  ArenaMatrix(Memory::Arena &arena, std::size_t rows, std::size_t cols)
      : m_Arena(&arena) {
    allocate(rows, cols);
  }

  // TODO: Add copy and move constructors and assignment operators
  ArenaMatrix(const ArenaMatrix &other) = delete;
  ArenaMatrix &operator=(const ArenaMatrix &other) = delete;
  ArenaMatrix(ArenaMatrix &&other) noexcept = default;
  ArenaMatrix &operator=(ArenaMatrix &&other) noexcept = default;

  void allocate(std::size_t rows, std::size_t cols) {
    if (!m_Arena)
      throw std::runtime_error("ArenaMatrix::allocate m_Arena = nullptr");
    m_Rows = rows;
    m_Cols = cols;
    m_Data = m_Arena->allocate<T>(m_Rows * m_Cols);
  }

  std::size_t rows() const noexcept { return m_Rows; }
  std::size_t cols() const noexcept { return m_Cols; }

  T *data() noexcept { return m_Data; }
  const T *data() const noexcept { return m_Data; }

  MatrixView<T> view() noexcept {
    return MatrixView<T>(m_Data, m_Rows, m_Cols,
                         static_cast<std::ptrdiff_t>(m_Cols), 1);
  }

  MatrixView<const T> cview() const noexcept {
    return MatrixView<const T>(m_Data, m_Rows, m_Cols,
                               static_cast<std::ptrdiff_t>(m_Cols), 1);
  }

  void fill_zeroes() { std::memset(m_Data, 0, m_Rows * m_Cols); }

private:
  Memory::Arena *m_Arena = nullptr;
  std::size_t m_Rows = 0, m_Cols = 0;
  T *m_Data = nullptr;
};

template <class T> class ScratchArenaMatrix {
public:
  ScratchArenaMatrix() = default;
  ScratchArenaMatrix(Memory::ScratchArena &arena, std::size_t rows,
                     std::size_t cols)
      : m_Arena(&arena), m_Rows(rows), m_Cols(cols) {
    allocate();
  }
  ~ScratchArenaMatrix() { release(); }

  ScratchArenaMatrix(const ScratchArenaMatrix &other) = delete;
  ScratchArenaMatrix &operator=(const ScratchArenaMatrix &other) = delete;

  ScratchArenaMatrix(ScratchArenaMatrix &&other) {
    if (*this == other)
      return;

    m_Arena = other.m_Arena;
    m_Rows = other.m_Rows;
    m_Cols = other.m_Cols;
    m_Offset = other.m_Offset;
    m_Data = other.m_Data;

    other.m_Arena = nullptr;
    other.m_Data = nullptr;
    other.m_Rows = other.m_Cols = other.m_Offset = 0;
  }

  ScratchArenaMatrix &operator=(ScratchArenaMatrix &&other) {
    if (*this == other)
      return *this;

    release();
    m_Arena = other.m_Arena;
    m_Rows = other.m_Rows;
    m_Cols = other.m_Cols;
    m_Offset = other.m_Offset;
    m_Data = other.m_Data;

    other.m_Arena = nullptr;
    other.m_Data = nullptr;
    other.m_Rows = other.m_Cols = other.m_Offset = 0;
  }

  void allocate() {
    if (!m_Arena)
      throw std::logic_error("ScratchArenaMatrix::allocate m_Arena = nullptr");
    if (m_Allocated)
      throw std::logic_error(
          "ScratchArenaMatrix::allocate should allocate only once");

    m_Offset = m_Arena->offset();
    m_Data = m_Arena->allocate<T>(m_Rows * m_Cols);
    m_Allocated = true;
  }

  void release() noexcept {
    if (m_Arena && m_Allocated)
      m_Arena->rewind(m_Offset);

    m_Arena = nullptr;
    m_Data = nullptr;
    m_Rows = m_Cols = m_Offset = 0;
    m_Allocated = false;
  }

  std::size_t rows() const noexcept { return m_Rows; }
  std::size_t cols() const noexcept { return m_Cols; }

  T *data() noexcept { return m_Data; }
  const T *data() const noexcept { return m_Data; }

  MatrixView<T> view() noexcept {
    return MatrixView<T>(m_Data, m_Rows, m_Cols,
                         static_cast<std::ptrdiff_t>(m_Cols), 1);
  }

  MatrixView<const T> cview() const noexcept {
    return MatrixView<const T>(m_Data, m_Rows, m_Cols,
                               static_cast<std::ptrdiff_t>(m_Cols), 1);
  }

  void fill_zeroes() { std::memset(m_Data, 0, m_Rows * m_Cols); }

private:
  Memory::ScratchArena *m_Arena = nullptr;
  std::size_t m_Rows = 0, m_Cols = 0, m_Offset = 0;
  bool m_Allocated = false;
  T *m_Data = nullptr;
};
} // namespace Logos::linalg
