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

constexpr std::size_t dtype_size(DType type) {
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
}

} // namespace ml::core
