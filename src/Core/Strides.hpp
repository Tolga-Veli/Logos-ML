#pragma once

#include <vector>

#include "Shape.hpp"

namespace ml::core {
class Strides {
public:
  Strides() = default;

  explicit Strides(std::vector<int> strides) : m_Strides(std::move(strides)) {}

  [[nodiscard]]
  static Strides Contiguous(const Shape &shape) {
    std::vector<int> strides(shape.rank());
    int stride = 1;

    for (int idx = shape.rank() - 1; idx >= 0; idx--) {
      strides[idx] = stride;
      stride *= shape[idx];
    }

    return Strides(std::move(strides));
  }

  [[nodiscard]]
  int operator[](int idx) const {
    return m_Strides[idx];
  }

  [[nodiscard]]
  int rank() const {
    return m_Strides.size();
  }

  [[nodiscard]]
  const std::vector<int> &values() const {
    return m_Strides;
  }

  [[nodiscard]] bool operator==(const Strides &other) const {
    return m_Strides == other.m_Strides;
  }

private:
  std::vector<int> m_Strides;
};
} // namespace ml::core
