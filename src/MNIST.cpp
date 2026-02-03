#include "MNIST.hpp"
#include "Math/Functions.hpp"

#include <fstream>
#include <iostream>
#include <vector>

namespace Logos::NeuralNet {

MNIST::MNIST(Memory::Arena &m_Arena)
    : m_TrainImgs(m_Arena, 60000, 784), m_TestImgs(m_Arena, 10000, 784),
      m_TrainLabels(load_labels_mat("data/train_labels.mat", 60000)),
      m_TestLabels(load_labels_mat("data/test_labels.mat", 10000)) {
  load_images_mat("data/train_images.mat", 60000, 28, 28, m_TrainImgs.view());
  load_images_mat("data/test_images.mat", 10000, 28, 28, m_TestImgs.view());
}

void MNIST::show_prediction(Memory::ScratchArena &scratchArena,
                            linalg::MatrixView<float> logits,
                            NeuralNetwork &model,
                            linalg::MatrixView<const float> imgs,
                            const std::vector<uint8_t> &labels,
                            std::size_t idx) {
  if (idx >= imgs.rows() || idx >= labels.size())
    throw std::logic_error("Show prediction: idx out of range");

  std::vector<float> img = get_mnist_image(imgs, idx);
  draw_mnist_digit(img);

  const auto D = imgs.cols();
  linalg::ScratchArenaMatrix<float> X(scratchArena, BATCH_SIZE, D);

  for (std::size_t i = 0; i < BATCH_SIZE; ++i)
    for (std::size_t j = 0; j < D; ++j)
      X.view()(i, j) = 0.0f;

  for (std::size_t j = 0; j < D; ++j)
    X.view()(0, j) = img[j];

  model.Forward(X.cview(), logits, false);

  std::cout << "\nPrediction: " << ArgmaxRow<float>(logits.cview(), 0)
            << " | Ground truth: " << static_cast<unsigned>(labels[idx])
            << "\n\n";
}

void MNIST::draw_mnist_digit(const std::vector<float> &data) {
  for (std::size_t y = 0; y < 28; y++) {
    for (std::size_t x = 0; x < 28; x++) {
      const float num = data[y * 28 + x];
      const std::uint32_t col = 232u + static_cast<std::uint32_t>(num * 23.0f);
      std::printf("\x1b[48;5;%dm  ", static_cast<unsigned>(col));
    }
    std::printf("\n");
  }
  std::printf("\x1b[0m");
}

std::vector<float> MNIST::get_mnist_image(linalg::MatrixView<const float> imgs,
                                          std::size_t idx) {
  if (idx >= imgs.rows())
    throw std::logic_error("Show prediction: idx out of range");

  const auto D = imgs.cols();
  std::vector<float> out(D);
  for (std::size_t j = 0; j < D; j++)
    out[j] = std::clamp(imgs(idx, j), 0.0f, 1.0f);
  return out;
}

void MNIST::load_images_mat(std::string path, std::size_t num, std::size_t rows,
                            std::size_t cols, linalg::MatrixView<float> out) {
  const auto D = rows * cols, total = num * D;
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open: " + path);

  in.read(reinterpret_cast<char *>(out.data()),
          static_cast<std::streamsize>(total * sizeof(float)));

  if (!in)
    throw std::runtime_error("Failed reading: " + path);
}

std::vector<uint8_t> MNIST::load_labels_mat(std::string path, std::size_t num) {

  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open: " + path);

  std::vector<std::uint8_t> labels(num);
  in.read(reinterpret_cast<char *>(labels.data()),
          static_cast<std::streamsize>(num));

  if (!in)
    throw std::runtime_error("Failed reading: " + path);

  return labels;
}

} // namespace Logos::NeuralNet
