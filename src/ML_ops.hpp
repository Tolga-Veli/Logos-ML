#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

namespace logos::func {

template <typename value_type = double>
inline std::vector<value_type> softmax(std::vector<value_type> logits) {
  value_type maxv = *std::max_element(logits.begin(), logits.end());

  value_type sum{};
  for (auto &val : logits) {
    val = std::exp(val - maxv);
    sum += val;
  }

  for (auto &val : logits)
    val /= sum;

  return logits;
}

template <typename value_type = double>
inline value_type cross_entropy(std::vector<value_type> &probs, int label) {
  const value_type eps = static_cast<value_type>(1e-12);
  value_type p = probs(static_cast<std::size_t>(label));
  if (p < eps)
    p = eps;
  return -std::log(p);
}

template <typename value_type = double>
inline std::size_t argmax(const std::vector<value_type> &v) {
  return static_cast<std::size_t>(
      std::distance(v.begin(), std::max_element(v.begin(), v.end())));
}

}; // namespace logos::func