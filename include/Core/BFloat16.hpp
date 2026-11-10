#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ml::core {
class BFloat16 {
public:
  constexpr BFloat16() noexcept = default;
  BFloat16(float val) noexcept {}

  BFloat16 &operator+=(BFloat16 other) noexcept {}
  BFloat16 &operator-=(BFloat16 other) noexcept {}
  BFloat16 &operator*=(BFloat16 other) noexcept {}
  BFloat16 &operator/=(BFloat16 other) noexcept {}

  BFloat16 operator+(BFloat16 other) const noexcept {
    BFloat16 val(0.0f);
    val += other;
    return val;
  }

  BFloat16 operator-(BFloat16 other) const noexcept {
    BFloat16 val(0.0f);
    val -= other;
    return val;
  }

  BFloat16 operator*(BFloat16 other) const noexcept {
    BFloat16 val(0.0f);
    val *= other;
    return val;
  }

  BFloat16 operator/(BFloat16 other) const noexcept {
    BFloat16 val(0.0f);
    val /= other;
    return val;
  }

  operator float() const noexcept {}

private:
  // 1 11111111 111
  // Sign - 1 bit
  // Exp - 8 bits
  // Mantissa - 7 bits

  std::uint16_t m_Data;
};
} // namespace ml::core

namespace std {
template <class T> ml::core::BFloat16 exp(ml::core::BFloat16 val) {}
template <class T> ml::core::BFloat16 log(ml::core::BFloat16 val) {}
template <class T> ml::core::BFloat16 sqrt(ml::core::BFloat16 val) {}
template <class T> ml::core::BFloat16 pow(ml::core::BFloat16 val) {}
template <class T> ml::core::BFloat16 abs(ml::core::BFloat16 val) {}

} // namespace std
