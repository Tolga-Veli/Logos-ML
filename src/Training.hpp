#pragma once

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "MLP.hpp"
#include "MNIST.hpp"
#include "Memory/Arena.hpp"

namespace Logos::NeuralNet {
class TrainModel {
public:
  TrainModel(Memory::Arena &m_Arena);
  void run();

private:
  Memory::Arena *m_Arena{nullptr};

  MLP_Hardcoded m_Model;
  float m_LearningRate = LEARNING_RATE;

  std::mt19937 m_RNG;
  std::vector<std::size_t> m_Order;

  MNIST m_datasetInfo;

  void make_batch(linalg::MatrixView<const float> imgs,
                  const std::vector<std::uint8_t> &labels,
                  const std::vector<std::size_t> &indices, std::size_t start,
                  std::size_t batch_size, linalg::MatrixView<float> Xb,
                  std::vector<std::uint8_t> &yb);
};

} // namespace Logos::NeuralNet
