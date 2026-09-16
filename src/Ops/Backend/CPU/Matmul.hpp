#pragma once

#include "BLAS.hpp"
#include "Ops/Utils.hpp"
#include "Ops/Views/MatrixView.hpp"

#include <cblas.h>

namespace ml::backend::cpu {
// C := alpha * op(A) * op(B) + beta * C   (written into `out`, i.e. C)
template <class T> void matmul(MatrixView<const T> A, ops::Transpose transA, MatrixView<const T> B,
                               ops::Transpose transB, T alpha, T beta, MatrixView<T> out) {

  CBLAS_TRANSPOSE opA, opB;
  std::size_t M, K, N;

  if (transA == ops::Transpose::Yes) {
    opA = CblasTrans;
    M = A.cols();
    K = A.rows();
  } else {
    opA = CblasNoTrans;
    M = A.rows();
    K = A.cols();
  }

  if (transB == ops::Transpose::Yes) {
    opB = CblasTrans;
    N = B.rows();
  } else {
    opB = CblasNoTrans;
    N = B.cols();
  }

  detail::BlasDispatch<T>::gemm(CblasRowMajor, opA, opB, M, N, K, alpha, A.data(), A.row_stride(), B.data(),
                                B.row_stride(), beta, out.data(), out.row_stride());
}

template <class T> void matmul(MatrixView<const T> A, MatrixView<const T> B, T alpha, T beta, MatrixView<T> out) {
  detail::BlasDispatch<T>::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(), A.cols(), alpha,
                                A.data(), A.row_stride(), B.data(), B.row_stride(), beta, out.data(), out.row_stride());
}

} // namespace ml::backend::cpu
