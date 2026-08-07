#pragma once

#include "Backend/BLAS.hpp"
#include "Core/Assert.hpp"
#include "Ops/Utils.hpp"
#include "Ops/Views/MatrixView.hpp"

#include <cblas.h>

namespace ml::kernels::cpu {
using ops::Transpose;
// C := alpha * op(A) * op(B) + beta * C   (written into `out`, i.e. C)
//
// op(A) is transA==Yes ? A^T : A,  shape M x K
// op(B) is transB==Yes ? B^T : B,  shape K x N
// out (C) must be shape M x N
//
// M, K come from A (adjusted for transA); N, K come from B (adjusted for
// transB) -- K must agree from both sides, checked by the assert below.
template <class T>
void matmul_kernel(Transpose transA, Transpose transB, T alpha,
                   MatrixView<const T> A, MatrixView<const T> B, T beta,
                   MatrixView<T> out) {
  const auto opA = (transA == Transpose::Yes ? CblasTrans : CblasNoTrans);
  const auto opB = (transB == Transpose::Yes ? CblasTrans : CblasNoTrans);

  const auto M = (transA == Transpose::Yes ? A.cols() : A.rows());
  const auto K = (transA == Transpose::Yes ? A.rows() : A.cols());
  const auto N = (transB == Transpose::Yes ? B.rows() : B.cols());

  // Shared dimension must match on both operands.
  CORE_VERIFY(K == (transB == Transpose::Yes ? B.cols() : B.rows()),
              "Invalid input matrix shape");

  // Output shape must match M x N derived above.
  CORE_VERIFY(out.rows() == M && out.cols() == N,
              "Invalid output matrix shape");

  detail::BlasDispatch<T>::gemm(CblasRowMajor, opA, opB, M, N, K, alpha,
                                A.data(), A.ld(), B.data(), B.ld(), beta,
                                out.data(), out.ld());
}

} // namespace ml::kernels::cpu
