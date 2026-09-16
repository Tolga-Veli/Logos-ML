#pragma once

#include "Core/Assert.hpp"

#include <cstdint>

namespace ml::core {
enum class DType : std::uint8_t {
  Float32,
  Float64,

  Int32,
};

template <DType> struct DTypeTraits {
  static_assert("Undefined dtype traits");
};

template <> struct DTypeTraits<DType::Float32> {
  using base_type = float;
};

template <> struct DTypeTraits<DType::Float64> {
  using base_type = double;
};

template <> struct DTypeTraits<DType::Int32> {
  using base_type = int;
};

template <DType D> using dtype_to_base_t = typename DTypeTraits<D>::base_type;

template <class T> struct BaseTypeTraits {
  static_assert("Undefined base type traits");
};

template <> struct BaseTypeTraits<float> {
  static constexpr DType dtype = DType::Float32;
};

template <> struct BaseTypeTraits<double> {
  static constexpr DType dtype = DType::Float64;
};

template <> struct BaseTypeTraits<int> {
  static constexpr DType dtype = DType::Int32;
};

template <class T> inline constexpr DType dtype_of_v = BaseTypeTraits<std::remove_cvref_t<T>>::dtype;

[[nodiscard]] constexpr std::string_view to_string(DType dtype) noexcept {
  switch (dtype) {
  case DType::Float32:
    return "Float32";
  case DType::Float64:
    return "Float64";
  case DType::Int32:
    return "Int32";
  }

  UNREACHABLE("Unknown dtype");
  return "Unknown";
}

[[nodiscard]] constexpr std::size_t dtype_size(DType type) {
  using enum DType;
  switch (type) {
  case Float32:
    return sizeof(float);
  case Float64:
    return sizeof(double);
  case Int32:
    return sizeof(int);
  }

  UNREACHABLE("Unknown dtype");
  return 0;
}

} // namespace ml::core

namespace std {
template <> struct formatter<ml::core::DType, char> : formatter<string_view, char> {
  template <class FormatContext> auto format(ml::core::DType dtype, FormatContext &ctx) const {
    return formatter<string_view, char>::format(ml::core::to_string(dtype), ctx);
  }
};
} // namespace std
