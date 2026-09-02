#pragma once

#include "Backend/BLAS.hpp"
#include "Ops/Utils.hpp"
#include "Ops/Views/MatrixView.hpp"

#include <cblas.h>

namespace ml::kernels::cpu {
// C := alpha * op(A) * op(B) + beta * C   (written into `out`, i.e. C)
template <class T>
void matmul(ops::Transpose transA, ops::Transpose transB, T alpha,
            MatrixView<const T> A, MatrixView<const T> B, T beta,
            MatrixView<T> out) {

  const auto opA = (transA == ops::Transpose::Yes ? CblasTrans : CblasNoTrans),
             opB = (transB == ops::Transpose::Yes ? CblasTrans : CblasNoTrans);

  const int M = (transA == ops::Transpose::Yes ? A.cols() : A.rows()),
            K = (transA == ops::Transpose::Yes ? A.rows() : A.cols()),
            N = (transB == ops::Transpose::Yes ? B.rows() : B.cols());

  detail::BlasDispatch<T>::gemm(CblasRowMajor, opA, opB, M, N, K, alpha,
                                A.data(), A.ld(), B.data(), B.ld(), beta,
                                out.data(), out.ld());
}

} // namespace ml::kernels::cpu
