#pragma once

#include <vector>

#include "Shape.hpp"

namespace ml::core {
class Strides {
public:
  Strides() = default;
  ~Strides() noexcept = default;

  explicit Strides(std::vector<std::size_t> strides) : m_Strides(std::move(strides)) {}

  [[nodiscard]] static Strides Contiguous(const Shape &shape) {
    std::vector<std::size_t> strides(shape.rank());
    std::size_t stride = 1;

    for (std::size_t idx = shape.rank(); idx-- > 0;) {
      strides[idx] = stride;
      stride *= shape[idx];
    }

    return Strides(std::move(strides));
  }

  [[nodiscard]] std::size_t rank() const { return m_Strides.size(); }
  [[nodiscard]] const std::vector<std::size_t> &values() const { return m_Strides; }

  [[nodiscard]] std::size_t operator[](int idx) const { return m_Strides[idx]; }
  [[nodiscard]] bool operator==(const Strides &other) const { return m_Strides == other.m_Strides; }

  void swap(std::size_t a, std::size_t b) { std::swap(m_Strides[a], m_Strides[b]); }

private:
  std::vector<std::size_t> m_Strides;
};
} // namespace ml::core
