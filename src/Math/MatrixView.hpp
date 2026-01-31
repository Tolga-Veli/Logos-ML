#pragma once
#include <cassert>
#include <cstddef>

namespace Logos {
// General strided matrix view (supports row-major, col-major, subviews,
// transpose)
template <class T> class MatrixView {
public:
  constexpr MatrixView() noexcept
      : m_Data(nullptr), m_Rows(0), m_Cols(0), m_RowStride(0), m_ColStride(0) {}

  constexpr MatrixView(T *data, std::size_t rows, std::size_t cols,
                       std::ptrdiff_t row_stride,
                       std::ptrdiff_t col_stride) noexcept
      : m_Data(data), m_Rows(rows), m_Cols(cols), m_RowStride(row_stride),
        m_ColStride(col_stride) {}

  constexpr T *data() const noexcept { return m_Data; }
  constexpr std::size_t rows() const noexcept { return m_Rows; }
  constexpr std::size_t cols() const noexcept { return m_Cols; }
  constexpr std::ptrdiff_t row_stride() const noexcept { return m_RowStride; }
  constexpr std::ptrdiff_t col_stride() const noexcept { return m_ColStride; }

  constexpr bool empty() const noexcept { return m_Rows == 0 || m_Cols == 0; }

  constexpr bool is_row_major_contiguous() const noexcept {
    return m_ColStride == 1 &&
           m_RowStride == static_cast<std::ptrdiff_t>(m_Cols);
  }

  constexpr bool is_col_major_contiguous() const noexcept {
    return m_RowStride == 1 &&
           m_ColStride == static_cast<std::ptrdiff_t>(m_Rows);
  }

  constexpr T &operator()(std::size_t r, std::size_t c) const noexcept {
    assert(r < m_Rows && c < m_Cols);
    return m_Data[static_cast<std::ptrdiff_t>(r) * m_RowStride +
                  static_cast<std::ptrdiff_t>(c) * m_ColStride];
  }

  static constexpr MatrixView RowMajor(T *data, std::size_t rows,
                                       std::size_t cols) noexcept {
    return MatrixView(data, rows, cols, static_cast<std::ptrdiff_t>(cols), 1);
  }

  static constexpr MatrixView ColMajor(T *data, std::size_t rows,
                                       std::size_t cols) noexcept {
    return MatrixView(data, rows, cols, 1, static_cast<std::ptrdiff_t>(rows));
  }

  constexpr MatrixView subview(std::size_t r0, std::size_t c0,
                               std::size_t rcount,
                               std::size_t ccount) const noexcept {
    assert(r0 + rcount <= m_Rows);
    assert(c0 + ccount <= m_Cols);

    T *p = m_Data + static_cast<std::ptrdiff_t>(r0) * m_RowStride +
           static_cast<std::ptrdiff_t>(c0) * m_ColStride;

    return MatrixView(p, rcount, ccount, m_RowStride, m_ColStride);
  }

  constexpr MatrixView transpose() const noexcept {
    return MatrixView(m_Data, m_Cols, m_Rows, m_ColStride, m_RowStride);
  }

private:
  T *m_Data;
  std::size_t m_Rows, m_Cols;
  std::ptrdiff_t m_RowStride, m_ColStride;
};

} // namespace Logos
