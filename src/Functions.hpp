#pragma once

#include <cmath>
#include <limits>
#include "Linalg.hpp"

namespace NeuralNet {
inline void Softmax(const linalg::Matrix &logits, linalg::Matrix &probs) {
  const auto N = logits.GetRows(), M = logits.GetCols();
  if (N == 0 || M == 0)
    throw std::logic_error("Softmax: empty logits");

  if (probs.GetRows() != N || probs.GetCols() != M)
    probs = linalg::Matrix(N, M);

  const auto X = logits.data();
  auto Z = probs.data();

  for (uint64_t i = 0; i < N; i++) {
    float maxv = std::numeric_limits<float>::lowest();
    for (uint64_t j = 0; j < M; j++)
      maxv = std::max(maxv, X[i * M + j]);

    float sum = 0.0f;
    for (uint64_t j = 0; j < M; j++) {
      float val = std::exp(X[i * M + j] - maxv);
      Z[i * M + j] = val;
      sum += val;
    }

    const float invSum = 1.0f / sum;
    for (uint64_t j = 0; j < M; j++)
      Z[i * M + j] *= invSum;
  }
}

inline float SoftmaxCrossEntropyFromLogits(const linalg::Matrix &logits,
                                           const std::vector<uint8_t> &labels,
                                           linalg::Matrix &dLogits) {
  const uint64_t N = logits.GetRows();
  const uint64_t M = logits.GetCols();
  if (N == 0 || M == 0 || labels.size() != N)
    throw std::logic_error("SoftmaxCrossEntropyFromLogits: wrong input");

  if (dLogits.GetRows() != N || dLogits.GetCols() != M)
    dLogits = linalg::Matrix(N, M);

  const float invN = 1.0f / static_cast<float>(N);
  float loss_sum = 0.0f;

  const float *X = logits.data();
  float *G = dLogits.data();

  for (uint64_t i = 0; i < N; ++i) {
    const uint8_t y = labels[i];
    if (y >= M)
      throw std::logic_error(
          "SoftmaxCrossEntropyFromLogits: label out of range");

    const float *row = X + i * M;
    float *grow = G + i * M;

    float maxv = row[0];
    for (uint64_t j = 1; j < M; ++j)
      maxv = std::max(maxv, row[j]);

    float sumExp = 0.0f;
    for (uint64_t j = 0; j < M; ++j)
      sumExp += std::exp(row[j] - maxv);

    const float logSumExp = maxv + std::log(sumExp);
    loss_sum += (-row[y] + logSumExp);

    const float invSumExp = 1.0f / sumExp;
    for (uint64_t j = 0; j < M; ++j) {
      const float p = std::exp(row[j] - maxv) * invSumExp;
      float g = p;
      if (j == static_cast<uint64_t>(y))
        g -= 1.0f;
      grow[j] = g * invN;
    }
  }

  return loss_sum * invN;
}

inline float CrossEntropy(const linalg::Matrix &probs,
                          const std::vector<uint8_t> &labels,
                          linalg::Matrix &dLogits) {
  const auto N = probs.GetRows(), M = probs.GetCols();
  if (N == 0 || M == 0 || (labels.size() != N))
    throw std::logic_error("CrossEntropy: wrong input");

  if (dLogits.GetRows() != N || dLogits.GetCols() != M)
    dLogits = linalg::Matrix(N, M);

  float loss_sum = 0.0f;
  const float invN = 1.0f / float(N), eps = 1e-12f;

  for (uint64_t i = 0; i < N; i++) {
    const auto label = labels[i];
    if (label >= M)
      throw std::logic_error("CrossEntropy: label out of range");

    float val = probs.at(i, label);
    if (val < eps)
      val = eps;

    loss_sum += -std::log(val);

    for (uint64_t j = 0; j < M; j++) {
      float g = probs.at(i, j) * invN;
      if (j == label)
        g -= invN;
      dLogits.at(i, j) = g;
    }
  }
  return loss_sum * invN;
}

inline uint64_t ArgmaxRow(const linalg::Matrix &A, uint64_t row) {
  const auto N = A.GetRows(), M = A.GetCols();
  if (row >= N || M == 0)
    throw std::logic_error("ArgmaxRow: out of bounds");

  const auto X = A.data() + row * M;

  uint64_t best = 0;
  float maxv = std::numeric_limits<float>::lowest();
  for (uint64_t j = 0; j < M; j++) {
    float val = X[j];
    if (val > maxv) {
      best = j;
      maxv = val;
    }
  }
  return best;
}
} // namespace NeuralNet