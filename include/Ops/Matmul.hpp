#pragma once

#pragma once

#include "Core/Assert.hpp"
#include "Memory/Device.hpp"

#include "Ops/Backend/CPU/Matmul.hpp"
#include "Ops/Backend/Dispatch.hpp"

#include "Ops/Utils.hpp"
#include "Ops/Validation.hpp"

namespace ml::ops {

inline void matmul(const Tensor &A, Transpose transA, const Tensor &B, Transpose transB, bool overwrite, Tensor &out) {
  require_same_device(A, B, out);
  require_same_dtype(A, B, out);

  dispatch(A.device(), A.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::matmul<T>(A, transA, B, transB, T{1}, overwrite ? T{0} : T{1}, out);
    else
      UNREACHABLE();
  });
}

inline void matmul(const Tensor &A, const Tensor &B, bool overwrite, Tensor &out) {
  require_same_dtype(A, B, out);
  require_same_device(A, B, out);

  dispatch(A.device(), A.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::matmul<T>(A, B, T{1}, overwrite ? T{0} : T{1}, out);
    else
      UNREACHABLE();
  });
}
} // namespace ml::ops
