#pragma once

#include "Assert.hpp"

#include <cstdint>

namespace ml::core {
enum class DType : std::uint8_t {
  Float32,
  Float64,
  Int32, // # FIX: doesn't support any operations yet (can just store data)
};

template <class T> constexpr DType dtype_of();

template <> constexpr DType dtype_of<float>() { return DType::Float32; }
template <> constexpr DType dtype_of<double>() { return DType::Float64; }
template <> constexpr DType dtype_of<int>() { return DType::Int32; }
// template <> constexpr DType dtype_of<std::int64_t>() { return DType::Int64; }

template <DType type> struct dtype_to_cpp;

template <> struct dtype_to_cpp<DType::Float32> {
  using type = float;
};

template <> struct dtype_to_cpp<DType::Float64> {
  using type = double;
};

template <> struct dtype_to_cpp<DType::Int32> {
  using type = int;
};

/*template <> struct dtype_to_cpp<DType::Int64> {
  using type = std::int64_t;
};*/

template <DType type> using dtype_to_cpp_v = typename dtype_to_cpp<type>::type;

constexpr std::size_t dtype_size(DType type) {
  using enum DType;
  switch (type) {
  case Float32:
    return sizeof(float);
  case Float64:
    return sizeof(double);
  case Int32:
    return sizeof(int);
    /*case Int64:
      return sizeof(std::int64_t);*/
  }

  CORE_VERIFY(false, "unreachable dtype");
}
} // namespace ml::core
