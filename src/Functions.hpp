#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Matrix.inl"

namespace Logos::NeuralNet {
template <class T>
inline void Softmax(const linalg::Matrix<T> &logits, linalg::Matrix<T> &probs) {

  const auto N = logits.rows(), M = logits.cols();
  if (N == 0 || M == 0)
    throw std::logic_error("Softmax: empty logits");

  if (probs.rows() != N || probs.cols() != M)
    probs = linalg::Matrix<T>(N, M);

  for (std::size_t i = 0; i < N; i++) {
    T maxv = std::numeric_limits<T>::lowest();
    for (std::size_t j = 0; j < M; j++)
      maxv = std::max(maxv, logits(i, j));

    T sum{0};
    for (std::size_t j = 0; j < M; j++) {
      const T val = std::exp(logits(i, j) - maxv);
      probs(i, j) = val;
      sum += val;
    }

    const T invSum = T{1} / sum;
    for (std::size_t j = 0; j < M; j++)
      probs(i, j) *= invSum;
  }
}

template <class T>
inline T CrossEntropy(const linalg::Matrix<T> &probs,
                      const std::vector<std::uint8_t> &labels,
                      linalg::Matrix<T> &dLogits) {

  const auto N = probs.rows(), M = probs.cols();
  if (N == 0 || M == 0 || labels.size() != N)
    throw std::logic_error("CrossEntropy: wrong input");

  if (dLogits.rows() != N || dLogits.cols() != M)
    dLogits = linalg::Matrix<T>(N, M);

  const T invN = T{1} / N;
  const T eps = T{1e-12};

  T loss_sum{0};
  for (std::size_t i = 0; i < N; i++) {
    const std::size_t y = static_cast<std::size_t>(labels[i]);
    if (y >= M)
      throw std::logic_error("CrossEntropy: label out of range");

    T p_y = probs(i, y);
    if (p_y < eps)
      p_y = eps;
    loss_sum += -std::log(p_y);

    for (std::size_t j = 0; j < M; j++) {
      T g = probs(i, j) * invN;
      if (j == y)
        g -= invN;
      dLogits(i, j) = g;
    }
  }

  return loss_sum * invN;
}

template <class T>
inline std::size_t ArgmaxRow(const linalg::Matrix<T> &A, std::size_t row) {
  const auto N = A.rows(), M = A.cols();
  if (row >= N || M == 0)
    throw std::logic_error("ArgmaxRow: out of bounds");

  std::size_t best = 0;
  T maxv = std::numeric_limits<T>::lowest();
  for (std::size_t j = 0; j < M; j++) {
    const T val = A(row, j);
    if (val > maxv) {
      maxv = val;
      best = j;
    }
  }
  return best;
}
} // namespace Logos::NeuralNet
