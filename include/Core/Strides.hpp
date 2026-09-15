#pragma once

#include <vector>

#include "Shape.hpp"

namespace ml::core {
class Strides {
public:
  Strides() = default;
  ~Strides() noexcept = default;

  explicit Strides(std::vector<std::uint32_t> strides) : m_Strides(std::move(strides)) {}

  [[nodiscard]] static Strides Contiguous(const Shape &shape) {
    std::vector<std::uint32_t> strides(shape.rank());
    std::uint32_t stride = 1;

    for (int idx = shape.rank() - 1; idx >= 0; idx--) {
      strides[idx] = stride;
      stride *= shape[idx];
    }

    return Strides(std::move(strides));
  }

  [[nodiscard]] int rank() const { return m_Strides.size(); }
  [[nodiscard]] const std::vector<std::uint32_t> &values() const { return m_Strides; }

  [[nodiscard]] int operator[](int idx) const { return m_Strides[idx]; }
  [[nodiscard]] bool operator==(const Strides &other) const { return m_Strides == other.m_Strides; }

  void swap(int a, int b) { std::swap(m_Strides[a], m_Strides[b]); }

private:
  std::vector<std::uint32_t> m_Strides;
};
} // namespace ml::core
