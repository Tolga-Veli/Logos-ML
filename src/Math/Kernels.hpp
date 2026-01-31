#pragma once

#include "MatrixView.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace Logos::linalg {

template <class T>
inline void matmul(MatrixView<const T> A, MatrixView<const T> B,
                   MatrixView<T> &out) {
  if (A.cols() != B.rows())
    throw std::logic_error("matmul shape mismatch");

  // Matrix A is (N x K)
  // Matrix B is (K x M)
  // out is (N x M)
  const auto N = A.rows(), K = A.cols(), M = B.cols();
  if (out.rows() != N || out.cols() != M)
    out = MatrixView<T>(N, M);
  else
    out.fill_zeroes();

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
inline void add_rowwise_bias(const std::vector<T> &b, MatrixView<T> &out) {
  if (b.size() != out.cols())
    throw std::logic_error("add_rowwise_bias: size mismatch");

  const auto N = out.rows(), M = out.cols();
  auto X = out.data();
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < M; j++)
      X[i * M + j] += b[j];
}

template <class T>
inline void sum_rows(MatrixView<const T> A, std::vector<T> &out) {
  out.assign(A.cols(), 0.0f);

  const auto N = A.rows(), M = A.cols();
  const auto X = A.data();
  for (std::size_t i = 0; i < N; i++)
    for (std::size_t j = 0; j < M; j++)
      out[j] += X[i * M + j];
}

template <class T>
inline void matmul_transposeA(MatrixView<const T> A, MatrixView<const T> B,
                              MatrixView<T> &out) {
  if (A.rows() != B.rows())
    throw std::logic_error("matmul_transposeA: mismatch");

  // A = [N x M]
  // B = [N x P]
  // out = [M x P]

  const auto N = A.rows(), M = A.cols(), P = B.cols();
  if (out.rows() != M || out.cols() != P)
    out = MatrixView<T>(M, P);
  else
    out.fill_zeroes();

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
inline void matmul_transposeB(MatrixView<const T> A, MatrixView<const T> B,
                              MatrixView<T> &out) {
  if (A.cols() != B.cols())
    throw std::logic_error("matmul_transposeB: mismatch");

  // A = [N x M]
  // B = [P x M]
  // out = [N x P]

  const auto N = A.rows(), M = A.cols(), P = B.rows();
  if (out.rows() != N || out.cols() != P)
    out = MatrixView<T>(N, P);

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
} // namespace Logos::linalg
