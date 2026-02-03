#pragma once
#include <cassert>
#include <cstddef>
#include <stdexcept>

namespace Logos ::linalg {
// General strided matrix view (supports row-major, col-major, subviews,
// transpose)
template <class T> class MatrixView {
public:
  constexpr MatrixView() noexcept = default;
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

  constexpr T &operator()(std::size_t r, std::size_t c) const {
    if (r >= m_Rows || c >= m_Cols)
      throw std::logic_error("MatrixView()::operator() out of range");
    return m_Data[static_cast<std::ptrdiff_t>(r) * m_RowStride +
                  static_cast<std::ptrdiff_t>(c) * m_ColStride];
  }

  // 0-based indexing
  constexpr MatrixView subview(std::size_t r0, std::size_t c0,
                               std::size_t rcount, std::size_t ccount) const {
    if (r0 + rcount > m_Rows || c0 + ccount > m_Cols)
      throw std::logic_error("MatrixView::subview out of range");

    T *p = m_Data + static_cast<std::ptrdiff_t>(r0) * m_RowStride +
           static_cast<std::ptrdiff_t>(c0) * m_ColStride;

    return MatrixView(p, rcount, ccount, m_RowStride, m_ColStride);
  }

  // 0-based indexing
  // Slice ranges: [r0, r1], [c0,c1] - last element inclusive
  constexpr MatrixView slice(std::size_t r0, std::size_t r1, std::size_t c0,
                             std::size_t c1) {
    if (r1 < r0 || c1 < c0)
      throw std::logic_error("slice: end < begin");
    return subview(r0, c0, r1 - r0 + 1, c1 - c0 + 1);
  }

  // 0-based indexing
  constexpr MatrixView slice_col(std::size_t col) const {
    return subview(0, col, m_Rows, 0);
  }

  // 0 based indexing
  constexpr MatrixView slice_row(std::size_t row) const {
    return subview(row, 0, 0, m_Cols);
  }

  constexpr MatrixView<const T> cview() const noexcept {
    return MatrixView<const T>(m_Data, m_Rows, m_Cols, m_RowStride,
                               m_ColStride);
  }

  constexpr MatrixView transpose() const noexcept {
    return MatrixView(m_Data, m_Cols, m_Rows, m_ColStride, m_RowStride);
  }

private:
  T *m_Data{nullptr};
  std::size_t m_Rows = 0, m_Cols = 0;
  std::ptrdiff_t m_RowStride = 0, m_ColStride = 0;
};

} // namespace Logos::linalg
