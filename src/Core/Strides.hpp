#pragma once

#include <vector>

#include "Shape.hpp"

namespace ml::core {
class Strides {
public:
  Strides() = default;

  explicit Strides(std::vector<size_t> strides)
      : m_Strides(std::move(strides)) {}

  [[nodiscard]]
  static Strides Contiguous(const Shape &shape) {
    std::vector<size_t> strides(shape.rank());
    size_t stride = 1;

    for (size_t idx = shape.rank(); idx-- > 0;) {
      strides[idx] = stride;
      stride *= shape[idx];
    }

    return Strides(std::move(strides));
  }

  [[nodiscard]]
  size_t operator[](size_t idx) const {
    return m_Strides[idx];
  }

  [[nodiscard]]
  size_t rank() const {
    return m_Strides.size();
  }

  [[nodiscard]]
  const std::vector<size_t> &values() const {
    return m_Strides;
  }

private:
  std::vector<size_t> m_Strides;
};
} // namespace ml::core
