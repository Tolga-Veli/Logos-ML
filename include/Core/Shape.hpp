#pragma once

#include <initializer_list>
#include <numeric>
#include <vector>

namespace ml::core {
class Shape {
public:
  Shape() = default;
  ~Shape() noexcept = default;

  Shape(std::initializer_list<std::size_t> dims) : m_Dims(dims) {}
  explicit Shape(std::vector<std::size_t> dims) : m_Dims(std::move(dims)) {}

  [[nodiscard]] std::size_t rank() const noexcept { return m_Dims.size(); }
  [[nodiscard]] bool empty() const noexcept { return m_Dims.empty(); }
  [[nodiscard]] const std::vector<std::size_t> &dims() const noexcept { return m_Dims; }

  [[nodiscard]] std::size_t num_elements() const noexcept {
    return std::accumulate(m_Dims.begin(), m_Dims.end(), 1uz, std::multiplies<>{});
  }

  [[nodiscard]] std::size_t operator[](int idx) const noexcept { return m_Dims[idx]; }
  [[nodiscard]] bool operator==(const Shape &other) const noexcept { return m_Dims == other.m_Dims; }
  [[nodiscard]] bool operator!=(const Shape &other) const noexcept { return !(*this == other); }

  void swap(std::size_t a, std::size_t b) { std::swap(m_Dims[a], m_Dims[b]); }

private:
  std::vector<std::size_t> m_Dims;
};
} // namespace ml::core
