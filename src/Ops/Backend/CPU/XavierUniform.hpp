#pragma once

#include <cmath>
#include <random>

#include "Ops/Views/MatrixView.hpp"

namespace ml::backend::cpu {
template <class T> inline void xavier_uniform(MatrixView<T> data) {
  const T lim = std::sqrt(T{6} / static_cast<T>(data.rows() + data.cols()));

  std::uniform_real_distribution<T> dist(-lim, lim);
  static std::mt19937 rng{std::random_device{}()};

  for (std::size_t i = 0; i < data.rows(); i++)
    for (std::size_t j = 0; j < data.cols(); j++)
      data(i, j) = dist(rng);
}
} // namespace ml::backend::cpu
