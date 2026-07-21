#pragma once

#include <initializer_list>
#include <numeric>
#include <vector>

namespace ml::core {
class Shape {
public:
  Shape() = default;
  Shape(std::initializer_list<size_t> dims) : m_Dims(dims) {}
  explicit Shape(std::vector<size_t> dims) : m_Dims(std::move(dims)) {}

  [[nodiscard]] size_t rank() const { return m_Dims.size(); }

  [[nodiscard]] size_t num_elements() const {
    return std::accumulate(m_Dims.begin(), m_Dims.end(), static_cast<size_t>(1),
                           std::multiplies<>{});
  }

  [[nodiscard]]
  bool empty() const {
    return m_Dims.empty();
  }

  [[nodiscard]]
  size_t operator[](size_t idx) const {
    return m_Dims[idx];
  }

  [[nodiscard]] const std::vector<size_t> &dims() const { return m_Dims; }

  [[nodiscard]] bool operator==(const Shape &other) const {
    return m_Dims == other.m_Dims;
  }

private:
  std::vector<size_t> m_Dims;
};
} // namespace ml::core
