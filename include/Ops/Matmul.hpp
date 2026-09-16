#pragma once

#pragma once

#include "Core/Assert.hpp"
#include "Memory/Device.hpp"
#include "Ops/Backend/CPU/Matmul.hpp"
#include "Ops/Backend/Dispatch.hpp"
#include "Ops/Utils.hpp"

namespace ml {
namespace ops {

inline void matmul(const Tensor &A, Transpose transA, const Tensor &B, Transpose transB, bool overwrite, Tensor &out) {
  dispatch(A.device(), A.dtype(), [&]<memory::DeviceType D, class T>() {
    if constexpr (D == memory::DeviceType::CPU)
      backend::cpu::matmul<T>(A, transA, B, transB, T{1}, overwrite ? T{0} : T{1}, out);
    else
      UNREACHABLE();
  });
}

inline void matmul(const Tensor &A, const Tensor &B, bool overwrite, Tensor &out) {
  matmul(A, Transpose::No, B, Transpose::No, overwrite, out);
}
} // namespace ops
} // namespace ml
