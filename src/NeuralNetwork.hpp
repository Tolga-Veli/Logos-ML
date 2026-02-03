#pragma once

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "Layers/Linear.hpp"
#include "Layers/ReLU.hpp"
#include "Memory/Arena.hpp"

namespace Logos::NeuralNet {
static constexpr std::uint32_t INPUT_LAYER = 784, HIDDEN = 256,
                               OUTPUT_LAYER = 10, BATCH_SIZE = 64, EPOCHS = 10;
static constexpr float LEARNING_RATE = 0.05f, LEARNING_RATE_DECAY = 0.95f;

class MLP_Hardcoded {
public:
  MLP_Hardcoded() = delete;
  MLP_Hardcoded(Memory::Arena &m_Arena, std::size_t in_dim,
                std::size_t hidden_dim, std::size_t num_classes,
                std::mt19937 &rng);

  float TrainStep(linalg::MatrixView<const float> X,
                  const std::vector<uint8_t> &labels, float learning_rate);

  void Forward(linalg::MatrixView<const float> X,
               linalg::MatrixView<float> out);
  void Backward();
  void GradientDescentStep(float learning_rate);

  float Accuracy(linalg::MatrixView<const float> X,
                 const std::vector<uint8_t> &labels);

private:
  Memory::Arena &m_Arena;
  Linear<float> fc1, fc2;
  ReLU<float> relu;

  linalg::ArenaMatrix<float> A, H, logits, probs, dA, dH, dLogits, dX;
};

class TrainModel {
public:
  using NeuralNetwork = MLP_Hardcoded;

  TrainModel();
  void run();

private:
  Memory::Arena m_Arena;

  NeuralNetwork m_Model;
  float m_LearningRate;

  linalg::ArenaMatrix<float> m_TrainImgs, m_TestImgs;
  std::vector<uint8_t> m_TrainLabels, m_TestLabels;

  std::mt19937 m_RNG{};
  std::vector<std::size_t> m_Order;

  void load_images_mat(std::string path, std::size_t num, std::size_t rows,
                       std::size_t cols, linalg::MatrixView<float> out);
  std::vector<std::uint8_t> load_labels_mat(std::string path, std::size_t num);

  void make_batch(linalg::MatrixView<const float> imgs,
                  const std::vector<std::uint8_t> &labels,
                  const std::vector<std::size_t> &indices, std::size_t start,
                  std::size_t batch_size, linalg::MatrixView<float> Xb,
                  std::vector<std::uint8_t> &yb);

  void show_prediction(NeuralNetwork &model,
                       linalg::MatrixView<const float> imgs,
                       const std::vector<std::uint8_t> &labels,
                       std::size_t idx);

  void draw_mnist_digit(const std::vector<float> &data);

  std::vector<float> get_mnist_image(linalg::MatrixView<const float> imgs,
                                     std::size_t idx);
};

} // namespace Logos::NeuralNet
