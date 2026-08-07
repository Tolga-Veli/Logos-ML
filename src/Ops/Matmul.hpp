#pragma once

#include "Core/Assert.hpp"
#include "Memory/Device.hpp"
#include "Ops/Kernels/CPU/Matmul.hpp"
#include "Ops/Kernels/Dispatch.hpp"
#include "Ops/Utils.hpp"

namespace ml::ops {

inline void matmul(Transpose transA, Transpose transB, bool overwrite,
                   const Tensor &A, const Tensor &B, Tensor &out) {
  CORE_VERIFY(A.rank() == 2 && B.rank() == 2 && out.rank() == 2,
              "Expected matrices");

  CORE_VERIFY(A.dtype() == B.dtype() && A.dtype() == out.dtype(),
              "Dtype mismatch");

  using kernels::MatrixView;

  kernels::dispatch(A.device(), A.dtype(),
                    [&]<memory::DeviceType D, class T>() {
                      if constexpr (D == memory::DeviceType::CPU)
                        kernels::cpu::matmul_kernel<T>(
                            transA, transB, T{1}, MatrixView<const T>(A),
                            MatrixView<const T>(B), overwrite ? T{0} : T{1},
                            MatrixView<T>(out));
                      else {
                        CORE_VERIFY(false, "not implemented yet");
                      }
                    });
}

} // namespace ml::ops
