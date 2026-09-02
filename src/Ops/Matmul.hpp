#pragma once

#include "Core/Assert.hpp"
#include "Memory/Device.hpp"
#include "Ops/Kernels/CPU/Matmul.hpp"
#include "Ops/Kernels/Dispatch.hpp"
#include "Ops/Utils.hpp"

namespace ml::ops {

/*
 * @brief Computes a matrix multiplication with optional transposition
 *
 * Computes:
 *
 *   out = alpha * op(A) * op(B) + beta * out
 *
 * where:
 *
 *   op(A) = A     if transA == No
 *   op(A) = A^T  if transA == Yes
 *
 *   op(B) = B    if transB == No
 *   op(B) = B^T  if transB == Yes
 *
 * The operation is performed using:
 *
 *   alpha = 1
 *
 * and:
 *
 *   beta = 0  if overwrite == true
 *   beta = 1  if overwrite == false
 *
 * @note If overwrite is true, the previous contents of 'out' are ignored.
 *       Otherwise, the existing contents of 'out' are added to the result.
 *
 * @param transA Whether A should be transposed before multiplication.
 * @param transB Whether B should be transposed before multiplication.
 * @param overwrite Whether to overwrite `out` instead of accumulating into it.
 * @param A First input matrix.
 * @param B Second input matrix.
 * @param out Output matrix.
 *
 */

inline void matmul(Transpose transA, Transpose transB, bool overwrite,
                   const Tensor &A, const Tensor &B, Tensor &out) {
  CORE_VERIFY(A.rank() == 2 && B.rank() == 2 && out.rank() == 2,
              "Requires matrices");

  CORE_VERIFY(A.dtype() == B.dtype() && A.dtype() == out.dtype(),
              "Requires all tensors to have the same dtype");

  CORE_VERIFY(A.device() == B.device() && A.device() == out.device(),
              "Requires all tensors to be on the same device");

  [[maybe_unused]] const int M = transA == Transpose::Yes ? A.shape()[1]
                                                          : A.shape()[0],
                             K_A = transA == Transpose::Yes ? A.shape()[0]
                                                            : A.shape()[1],
                             K_B = transB == Transpose::Yes ? B.shape()[1]
                                                            : B.shape()[0],
                             N = transB == Transpose::Yes ? B.shape()[0]
                                                          : B.shape()[1];

  CORE_VERIFY(K_A == K_B, "Matmul inner dimensions must match");

  CORE_VERIFY(out.shape()[0] == M && out.shape()[1] == N,
              "Matmul output must have shape [M, N]");

  using kernels::MatrixView;

  kernels::dispatch(
      A.device(), A.dtype(), [&]<memory::DeviceType D, class T>() {
        if constexpr (D == memory::DeviceType::CPU)
          kernels::cpu::matmul<T>(transA, transB, T{1}, MatrixView<const T>(A),
                                  MatrixView<const T>(B),
                                  overwrite ? T{0} : T{1}, MatrixView<T>(out));
        else {
          CORE_VERIFY(false, "not implemented yet");
        }
      });
}

} // namespace ml::ops
