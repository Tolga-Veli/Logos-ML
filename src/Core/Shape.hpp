#pragma once

#include <initializer_list>
#include <numeric>
#include <vector>

namespace ml::core {
class Shape {
public:
  Shape() = default;
  Shape(std::initializer_list<int> dims) noexcept : m_Dims(dims) {}
  explicit Shape(std::vector<int> dims) noexcept : m_Dims(std::move(dims)) {}

  [[nodiscard]] int rank() const noexcept { return m_Dims.size(); }

  [[nodiscard]] int num_elements() const noexcept {
    return std::accumulate(m_Dims.begin(), m_Dims.end(), 1,
                           std::multiplies<>{});
  }

  [[nodiscard]]
  bool empty() const noexcept {
    return m_Dims.empty();
  }

  [[nodiscard]]
  int operator[](int idx) const noexcept {
    return m_Dims[idx];
  }

  [[nodiscard]] const std::vector<int> &dims() const noexcept { return m_Dims; }

  [[nodiscard]] bool operator==(const Shape &other) const noexcept {
    return m_Dims == other.m_Dims;
  }

  [[nodiscard]] bool operator!=(const Shape &other) const noexcept {
    return !(*this == other);
  }

private:
  std::vector<int> m_Dims;
};
} // namespace ml::core
