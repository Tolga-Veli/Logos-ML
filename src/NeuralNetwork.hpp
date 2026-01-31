#pragma once

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "Linear.hpp"
#include "ReLU.hpp"

namespace Logos::NeuralNet {
using Matrix = linalg::Matrix<float>;

class MLP_Hardcoded {
public:
  MLP_Hardcoded() = default;
  MLP_Hardcoded(std::size_t in_dim, std::size_t hidden_dim,
                std::size_t num_classes, std::mt19937 &rng);

  float TrainStep(const Matrix &X, const std::vector<uint8_t> &labels,
                  float learning_rate);
  void Forward(const Matrix &X, Matrix &out);
  float Accuracy(const Matrix &X, const std::vector<uint8_t> &labels);

private:
  Linear<float> fc1, fc2;
  ReLU<float> relu;

  Matrix A1, H1, logits, probs, dA1, dH1, dLogits, dX;
};

class TrainModel {
public:
  using NeuralNetwork = MLP_Hardcoded;

  TrainModel();
  void run();

private:
  static constexpr std::uint32_t INPUT_LAYER = 784, HIDDEN = 256,
                                 OUTPUT_LAYER = 10, BATCH_SIZE = 64,
                                 EPOCHS = 10;
  static constexpr float LEARNING_RATE = 0.05f, LEARNING_RATE_DECAY = 0.95f;

  NeuralNetwork m_Model;
  float m_LearningRate;

  Matrix m_TrainImgs, m_TestImgs;
  std::vector<uint8_t> m_TrainLabels, m_TestLabels;

  std::mt19937 m_RNG{};
  std::vector<std::size_t> m_Order;

  Matrix load_images_mat(std::string path, std::size_t num, std::size_t rows,
                         std::size_t cols);
  std::vector<std::uint8_t> load_labels_mat(std::string path, std::size_t num);

  void make_batch(const Matrix &imgs, const std::vector<std::uint8_t> &labels,
                  const std::vector<std::size_t> &indices, std::size_t start,
                  std::size_t batch_size, Matrix &Xb,
                  std::vector<std::uint8_t> &yb);

  void show_prediction(NeuralNetwork &model, const Matrix &imgs,
                       const std::vector<std::uint8_t> &labels,
                       std::size_t idx);
  void draw_mnist_digit(const std::vector<float> &data);
  std::vector<float> get_mnist_image(const Matrix &imgs, std::size_t idx);
};

} // namespace Logos::NeuralNet
