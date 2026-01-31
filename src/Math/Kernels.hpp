#pragma once

#include "Matrix.hpp"
#include "MatrixView.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace Logos::linalg {

template <class T>
inline void matmul_rowmajor(MatrixView<const T> A, MatrixView<const T> B,
                            MatrixView<T> out) {
  if (A.cols() != B.rows() || out.rows() != A.rows() || out.cols() != B.cols())
    throw std::logic_error("matmul_rowmajor: shape mismatch");

  if (!A.is_row_major_contiguous() || !B.is_row_major_contiguous() ||
      !out.is_row_major_contiguous())
    throw std::logic_error("matmul_rowmajor: requires row-major contiguous");

  const auto N = A.rows(), K = A.cols(), M = B.cols();
  const auto X = A.data(), Y = B.data();
  auto Z = out.data();
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < K; j++) {
      const T val = X[i * K + j];
      for (std::size_t k = 0; k < M; k++)
        Z[i * M + k] += val * Y[j * M + k];
    }
}

template <class T>
inline void matmul_colmajor(MatrixView<const T> A, MatrixView<const T> B,
                            MatrixView<T> out) {
  if (A.cols() != B.rows() || out.rows() != A.rows() || out.cols() != B.cols())
    throw std::logic_error("matmul_colmajor: shape mismatch");

  if (!A.is_col_major_contiguous() || !B.is_col_major_contiguous() ||
      !out.is_col_major_contiguous())
    throw std::logic_error("matmul_colmajor: requires col-major contiguous");

  const auto N = A.rows(), K = A.cols(), M = B.cols();
  const auto X = A.data(), Y = B.data();
  auto Z = out.data();
  for (std::size_t k = 0; k < M; k++)
    for (std::size_t j = 0; j < K; j++) {
      const T val = Y[k * M + j];
      for (std::size_t i = 0; i < N; i++)
        Z[k * N + i] += val * X[j * N + i];
    }
}

template <class T>
inline void matmul(MatrixView<const T> A, MatrixView<const T> B,
                   MatrixView<T> out) {
  if (A.cols() != B.rows() || out.rows() != A.rows() || out.cols() != B.cols())
    throw std::logic_error("matmul: shape mismatch");

  if (A.is_row_major_contiguous() && B.is_row_major_contiguous() &&
      out.is_row_major_contiguous())
    matmul_rowmajor<T>(A, B, out);

  else if (A.is_col_major_contiguous() && B.is_col_major_contiguous() &&
           out.is_col_major_contiguous())
    matmul_colmajor<T>(A, B, out);
  else
    throw std::logic_error("matmul: unsupported layout");
}

template <class T>
inline void matmul(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &out) {
  if (A.cols() != B.rows())
    throw std::logic_error("matmul: shape mismatch");

  const auto N = A.rows(), M = B.cols();
  if (out.rows() != N || out.cols() != M)
    out = Matrix<T>(N, M);
  else
    out.fill_zeroes();

  matmul<T>(A.view(), B.view(), out.view());
}

template <class T>
inline void add_rowwise_bias(const std::vector<T> &b, MatrixView<T> out) {
  if (b.size() != out.cols())
    throw std::logic_error("add_rowwise_bias: size mismatch");

  if (!out.is_row_major_contiguous())
    throw std::logic_error("Haven't implemented col-major kernels yet");

  const auto N = out.rows(), M = out.cols();
  auto X = out.data();
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < M; j++)
      X[i * M + j] += b[j];
}

template <class T>
inline void add_rowwise_bias(const std::vector<T> &b, Matrix<T> &out) {
  add_rowwise_bias<T>(b, out.view());
}

template <class T>
inline void sum_rows(MatrixView<const T> A, std::vector<T> &out) {
  if (!A.is_row_major_contiguous())
    throw std::logic_error("Haven't implemented col-major kernels yet");

  out.assign(A.cols(), 0.0f);
  const auto N = A.rows(), M = A.cols();
  const auto X = A.data();
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < M; j++)
      out[j] += X[i * M + j];
}

template <class T>
inline void sum_rows(const Matrix<T> &A, std::vector<T> &out) {
  sum_rows<T>(A.view(), out);
}

template <class T>
inline void matmul_transposeA(MatrixView<const T> A, MatrixView<const T> B,
                              MatrixView<T> out) {
  if (A.rows() != B.rows() || out.rows() != A.cols() || out.cols() != B.cols())
    throw std::logic_error("matmul_transposeA: shape mismatch");
  if (!A.is_row_major_contiguous() || !B.is_row_major_contiguous() ||
      !out.is_row_major_contiguous())
    throw std::logic_error(
        "matmul_transposeA: requires row-major contiguous views");

  // A = [N x M]
  // B = [N x P]
  // out = [M x P]

  const auto N = A.rows(), M = A.cols(), P = B.cols();
  const auto X = A.data(), Y = B.data();
  auto Z = out.data();

  for (std::size_t i = 0; i < M; i++)
    for (std::size_t k = 0; k < N; k++) {
      const auto val = X[k * M + i];
      for (std::size_t j = 0; j < P; j++)
        Z[i * P + j] += val * Y[k * P + j];
    }
}

template <class T>
inline void matmul_transposeA(const Matrix<T> &A, const Matrix<T> &B,
                              Matrix<T> &out) {
  if (A.rows() != B.rows())
    throw std::logic_error("matmul_transposeA: shape mismatch");

  const auto M = A.cols(), P = B.cols();
  if (out.rows() != M || out.cols() != P)
    out = Matrix<T>(M, P);
  else
    out.fill_zeroes();

  matmul_transposeA<T>(A.view(), B.view(), out.view());
}

template <class T>
inline void matmul_transposeB(MatrixView<const T> A, MatrixView<const T> B,
                              MatrixView<T> out) {
  if (A.cols() != B.cols() || out.rows() != A.rows() || out.cols() != B.rows())
    throw std::logic_error("matmul_transposeB: shape mismatch");

  if (!A.is_row_major_contiguous() || !B.is_row_major_contiguous() ||
      !out.is_row_major_contiguous())
    throw std::logic_error(
        "matmul_transposeB: requires row-major contiguous views");

  // A = [N x M]
  // B = [P x M]
  // out = [N x P]

  const auto N = A.rows(), M = A.cols(), P = B.rows();
  const auto X = A.data(), Y = B.data();
  auto Z = out.data();
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < P; j++) {
      T sum{0};
      for (std::size_t k = 0; k < M; k++)
        sum += X[i * M + k] * Y[j * M + k];
      Z[i * P + j] = sum;
    }
}

template <class T>
inline void matmul_transposeB(const Matrix<T> &A, const Matrix<T> &B,
                              Matrix<T> &out) {
  if (A.cols() != B.cols())
    throw std::logic_error("matmul_transposeB: shape mismatch");

  const auto N = A.rows(), P = B.rows();
  if (out.rows() != N || out.cols() != P)
    out = Matrix<T>(N, P);

  matmul_transposeB<T>(A.view(), B.view(), out.view());
}

} // namespace Logos::linalg
