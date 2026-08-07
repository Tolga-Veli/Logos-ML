#pragma once

#include "DType.hpp"

namespace ml::core {
template <class F> decltype(auto) dispatch_dtype(DType dtype, F &&f) {
  using enum DType;
  switch (dtype) {
  case Float32:
    return f.template operator()<float>();

  case Float64:
    return f.template operator()<double>();

  case Int32:
    return f.template operator()<int32_t>();
  }

  CORE_VERIFY(false, "unreachable");
}
} // namespace ml::core
