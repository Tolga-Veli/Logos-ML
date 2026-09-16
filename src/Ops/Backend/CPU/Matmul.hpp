#pragma once

#include "BLAS.hpp"
#include "Ops/Utils.hpp"
#include "Ops/Views/MatrixView.hpp"

#include <cblas.h>

namespace ml::backend::cpu {
// C := alpha * op(A) * op(B) + beta * C   (written into `out`, i.e. C)
template <class T> void matmul(MatrixView<const T> A, ops::Transpose transA, MatrixView<const T> B,
                               ops::Transpose transB, T alpha, T beta, MatrixView<T> out) {
  const auto opA = (transA == ops::Transpose::Yes ? CblasTrans : CblasNoTrans),
             opB = (transB == ops::Transpose::Yes ? CblasTrans : CblasNoTrans);

  const auto M = (transA == ops::Transpose::Yes ? A.cols() : A.rows()),
             K = (transA == ops::Transpose::Yes ? A.rows() : A.cols()),
             N = (transB == ops::Transpose::Yes ? B.rows() : B.cols());

  if (A.is_row_major() && B.is_row_major() && out.is_row_major())
    detail::BlasDispatch<T>::gemm(CblasRowMajor, opA, opB, M, N, K, alpha, A.data(), A.row_stride(), B.data(),
                                  B.row_stride(), beta, out.data(), out.row_stride());
  else if (A.is_col_major() && B.is_col_major() && out.is_col_major())
    detail::BlasDispatch<T>::gemm(CblasColMajor, opA, opB, M, N, K, alpha, A.data(), A.col_stride(), B.data(),
                                  B.col_stride(), beta, out.data(), out.col_stride());
}

} // namespace ml::backend::cpu
