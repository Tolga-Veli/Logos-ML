#pragma once

#include "Matrix.hpp"

#include <cstdint>
#include <stdexcept>

namespace linalg {

inline void matmul(const Matrix &A, const Matrix &B, Matrix &out) {
  if (A.GetCols() != B.GetRows())
    throw std::logic_error("matmul shape mismatch");

  // Matrix A is (N x K)
  // Matrix B is (K x M)
  // out is (N x M)
  const auto N = A.GetRows(), K = A.GetCols(), M = B.GetCols();
  if (out.GetRows() != N || out.GetCols() != M)
    out = Matrix(N, M);
  else
    out.Fill(0.0f);

  const auto X = A.data(), Y = B.data();
  auto Z = out.data();

  for (uint64_t i = 0; i < N; i++)
    for (uint64_t j = 0; j < K; j++) {
      const float val = X[i * K + j];
      for (uint64_t k = 0; k < M; k++)
        Z[i * M + k] += val * Y[j * M + k];
    }
}

inline void add_rowwise_bias(Matrix &out, const std::vector<float> &b) {
  if (b.size() != out.GetCols())
    throw std::logic_error("add_rowwise_bias: size mismatch");

  const auto N = out.GetRows(), M = out.GetCols();
  auto X = out.data();
  for (uint64_t i = 0; i < N; i++)
    for (uint64_t j = 0; j < M; j++)
      X[i * M + j] += b[j];
}

inline void sum_rows(const Matrix &A, std::vector<float> &vec) {
  vec.assign(A.GetCols(), 0.0f);

  const auto N = A.GetRows(), M = A.GetCols();
  const auto X = A.data();
  for (uint64_t i = 0; i < N; i++)
    for (uint64_t j = 0; j < M; j++)
      vec[j] += X[i * M + j];
}

inline void matmul_transposeA(const Matrix &A, const Matrix &B, Matrix &out) {
  if (A.GetRows() != B.GetRows())
    throw std::logic_error("matmul_transposeA: mismatch");

  // A = [N x M]
  // B = [N x P]
  // out = [M x P]

  const auto N = A.GetRows(), M = A.GetCols(), P = B.GetCols();
  if (out.GetRows() != M || out.GetCols() != P)
    out = Matrix(M, P);
  else
    out.Fill(0.0f);

  const auto X = A.data(), Y = B.data();
  auto Z = out.data();

  for (uint64_t i = 0; i < M; i++)
    for (uint64_t k = 0; k < N; k++) {
      const auto val = X[k * M + i];
      for (uint64_t j = 0; j < P; j++)
        Z[i * P + j] += val * Y[k * P + j];
    }
}

inline void matmul_transposeB(const Matrix &A, const Matrix &B, Matrix &out) {
  if (A.GetCols() != B.GetCols())
    throw std::logic_error("matmul_transposeB: mismatch");

  // A = [N x M]
  // B = [P x M]
  // out = [N x P]

  const auto N = A.GetRows(), M = A.GetCols(), P = B.GetRows();
  if (out.GetRows() != N || out.GetCols() != P)
    out = Matrix(N, P);

  const auto X = A.data(), Y = B.data();
  auto Z = out.data();

  for (uint64_t i = 0; i < N; i++)
    for (uint64_t j = 0; j < P; j++) {
      float sum = 0.0f;
      for (uint64_t k = 0; k < M; k++)
        sum += X[i * M + k] * Y[j * M + k];
      Z[i * P + j] = sum;
    }
}
} // namespace linalg