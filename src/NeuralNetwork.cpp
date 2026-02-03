#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "Math/Functions.hpp"
#include "NeuralNetwork.hpp"

namespace Logos::NeuralNet {
MLP_Hardcoded::MLP_Hardcoded(Memory::Arena &arena, std::size_t in_dim,
                             std::size_t hidden_dim, std::size_t num_classes,
                             std::mt19937 &rng)
    : m_Arena(arena), fc1(m_Arena, in_dim, hidden_dim, rng),
      fc2(m_Arena, hidden_dim, num_classes, rng),
      A(m_Arena, BATCH_SIZE, hidden_dim), H(m_Arena, BATCH_SIZE, hidden_dim),
      logits(m_Arena, BATCH_SIZE, num_classes),
      probs(m_Arena, BATCH_SIZE, num_classes),
      dA(m_Arena, BATCH_SIZE, hidden_dim), dH(m_Arena, BATCH_SIZE, hidden_dim),
      dLogits(m_Arena, BATCH_SIZE, num_classes),
      dX(m_Arena, BATCH_SIZE, in_dim) {}

float MLP_Hardcoded::TrainStep(linalg::MatrixView<const float> X,
                               const std::vector<uint8_t> &labels,
                               float learning_rate) {
  const auto N = X.rows(), M = X.cols();
  if (N == 0 || M == 0)
    throw std::logic_error("TrainStep: empty input matrix");
  if (labels.size() != N)
    throw std::logic_error("TrainStep: labels size mismatch");
  if (N != BATCH_SIZE)
    throw std::logic_error("TrainStep: X.rows() must equal BATCH_SIZE");

  // Forward Pass :Linear -> ReLU -> Linear -> Softmax -> CrossEntropy
  // Backward Pass : Linear -> ReLU-> Linear

  Forward(X, logits.view());

  Memory::ScratchArena scratchArena(m_Arena);
  linalg::ScratchArenaMatrix<float> probs(scratchArena, logits.rows(),
                                          logits.cols());
  Softmax<float>(logits.cview(), probs.view());
  const float loss = CrossEntropy<float>(probs.cview(), labels, dLogits.view());

  Backward();
  GradientDescentStep(learning_rate);
  return loss;
}

void MLP_Hardcoded::Forward(linalg::MatrixView<const float> X,
                            linalg::MatrixView<float> out) {
  if (X.rows() != BATCH_SIZE)
    throw std::logic_error("Forward: X.rows() must be BATCH_SIZE");
  if (out.rows() != BATCH_SIZE)
    throw std::logic_error("Forward: out.rows() must be BATCH_SIZE");

  fc1.Forward(X, A.view());
  relu.Forward(A.cview(), H.view());
  fc2.Forward(H.cview(), out);
}

void MLP_Hardcoded::Backward() {
  fc2.Backward(dLogits.cview(), dH.view());
  relu.Backward(dH.cview(), dA.view());
  fc1.Backward(dA.cview(), dX.view());
}

void MLP_Hardcoded::GradientDescentStep(float learning_rate) {
  fc1.GradientDescentStep(learning_rate);
  fc2.GradientDescentStep(learning_rate);

  fc1.ZeroGrads();
  fc2.ZeroGrads();
}

float MLP_Hardcoded::Accuracy(linalg::MatrixView<const float> X,
                              const std::vector<uint8_t> &labels) {
  Forward(X, logits.view());
  const auto N = logits.rows();
  if (N != labels.size() || N == 0)
    throw std::logic_error("MLP_Hardcoded::Accuracy wrong Matrix size");

  std::uint32_t cnt = 0;
  for (std::size_t i = 0; i < N; i++) {
    std::size_t pred = ArgmaxRow<float>(logits, i);
    if (pred == labels[i])
      cnt++;
  }

  return static_cast<float>(cnt) / N;
}

TrainModel::TrainModel()
    : m_Arena(1 * Memory::GiB),
      m_Model(m_Arena, INPUT_LAYER, HIDDEN, OUTPUT_LAYER, m_RNG),
      m_LearningRate(LEARNING_RATE), m_TrainImgs(m_Arena, 60000, 784),
      m_TestImgs(m_Arena, 10000, 784),
      m_TrainLabels(load_labels_mat("data/train_labels.mat", 60000)),
      m_TestLabels(load_labels_mat("data/test_labels.mat", 10000)) {

  load_images_mat("data/train_images.mat", 60000, 28, 28, m_TrainImgs.view());
  load_images_mat("data/test_images.mat", 10000, 28, 28, m_TestImgs.view());

  m_Order.resize(m_TrainImgs.rows());
  std::iota(m_Order.begin(), m_Order.end(), 0);

  std::cout << "Train: N=" << m_TrainImgs.rows()
            << " | Test: N=" << m_TestImgs.rows() << '\n';
}

void TrainModel::run() {
  std::vector<uint8_t> yb;

  for (std::uint32_t ep = 1; ep <= EPOCHS; ep++) {
    Memory::ScratchArena scratchArena(m_Arena);
    linalg::ScratchArenaMatrix<float> Xb(scratchArena, BATCH_SIZE,
                                         m_TrainImgs.cols());

    std::shuffle(m_Order.begin(), m_Order.end(), m_RNG);

    float loss_acc = 0.0;
    std::size_t steps = 0;

    const std::size_t sz = (m_Order.size() / BATCH_SIZE) * BATCH_SIZE;
    for (std::size_t start = 0; start < sz; start += BATCH_SIZE) {
      make_batch(m_TrainImgs.cview(), m_TrainLabels, m_Order, start, BATCH_SIZE,
                 Xb.view(), yb);

      const float loss = m_Model.TrainStep(Xb.cview(), yb, m_LearningRate);
      loss_acc += loss;
      steps++;

      if (steps % 500 == 0) {
        static std::uniform_int_distribution<std::size_t> dist_idx(
            0, yb.size() - 1);
        show_prediction(m_Model, m_TrainImgs.cview(), m_TrainLabels,
                        m_Order[start + dist_idx(m_RNG)]);
      }
    }

    std::uint32_t correct = 0, total = 0;
    linalg::ScratchArenaMatrix<float> Xt(scratchArena, BATCH_SIZE,
                                         m_TestImgs.cols()),
        logits(scratchArena, BATCH_SIZE, OUTPUT_LAYER);
    std::vector<uint8_t> yt;

    const std::size_t test_sz = (m_TestImgs.rows() / BATCH_SIZE) * BATCH_SIZE;
    for (std::size_t start = 0; start < test_sz; start += BATCH_SIZE) {
      std::vector<std::size_t> test_idx(BATCH_SIZE);
      std::iota(test_idx.begin(), test_idx.end(), start);

      make_batch(m_TestImgs.cview(), m_TestLabels, test_idx, 0, BATCH_SIZE,
                 Xt.view(), yt);

      m_Model.Forward(Xt.cview(), logits.view());

      for (std::size_t i = 0; i < BATCH_SIZE; i++) {
        const std::size_t pred = ArgmaxRow<float>(logits.cview(), i);
        if (pred == yt[i])
          correct++;
        total++;
      }
    }

    const float test_acc =
        (total == 0) ? 0.0f : static_cast<float>(correct) / total;
    const float mean_loss = (steps == 0) ? 0.0f : loss_acc / steps;

    std::cout << "Epoch " << ep << " done | lr=" << m_LearningRate
              << " mean_loss=" << mean_loss << " test_acc=" << test_acc << '\n';

    m_LearningRate *= LEARNING_RATE_DECAY;
  }
}

void TrainModel::load_images_mat(std::string path, std::size_t num,
                                 std::size_t rows, std::size_t cols,
                                 linalg::MatrixView<float> out) {
  const auto D = rows * cols, total = num * D;
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open: " + path);

  in.read(reinterpret_cast<char *>(out.data()),
          static_cast<std::streamsize>(total * sizeof(float)));

  if (!in)
    throw std::runtime_error("Failed reading: " + path);
}

std::vector<uint8_t> TrainModel::load_labels_mat(std::string path,
                                                 std::size_t num) {

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

void TrainModel::make_batch(linalg::MatrixView<const float> imgs,
                            const std::vector<std::uint8_t> &labels,
                            const std::vector<std::size_t> &indices,
                            std::size_t start, std::size_t batch_size,
                            linalg::MatrixView<float> Xb,
                            std::vector<std::uint8_t> &yb) {

  const auto D = imgs.cols(), N = indices.size(),
             end = std::min(start + batch_size, N), B = end - start;

  if (B == 0)
    throw std::logic_error("make_batch: empty batch");

  if (Xb.rows() < B || Xb.cols() != D)
    throw std::logic_error("make_batch: Xb has wrong shape");

  yb.resize(B);

  for (std::size_t i = 0; i < B; i++) {
    const auto idx = indices[start + i];
    if (idx >= imgs.rows() || idx >= labels.size())
      throw std::logic_error("make_batch: index out of range");

    yb[i] = labels[idx];

    for (std::size_t j = 0; j < D; j++)
      Xb(i, j) = imgs(idx, j);
  }
}

void TrainModel::show_prediction(NeuralNetwork &model,
                                 linalg::MatrixView<const float> imgs,
                                 const std::vector<uint8_t> &labels,
                                 std::size_t idx) {
  if (idx >= imgs.rows() || idx >= labels.size())
    throw std::logic_error("Show prediction: idx out of range");

  std::vector<float> img = get_mnist_image(imgs, idx);
  draw_mnist_digit(img);

  const auto D = imgs.cols();
  Memory::ScratchArena scratchArena(m_Arena);

  linalg::ScratchArenaMatrix<float> X(scratchArena, BATCH_SIZE, D);
  linalg::ScratchArenaMatrix<float> logits(scratchArena, BATCH_SIZE,
                                           OUTPUT_LAYER);

  for (std::size_t i = 0; i < BATCH_SIZE; ++i)
    for (std::size_t j = 0; j < D; ++j)
      X.view()(i, j) = 0.0f;

  for (std::size_t j = 0; j < D; ++j)
    X.view()(0, j) = img[j];

  model.Forward(X.cview(), logits.view());

  std::cout << "\nPrediction: " << ArgmaxRow<float>(logits.cview(), 0)
            << " | Ground truth: " << static_cast<unsigned>(labels[idx])
            << "\n\n";
}

void TrainModel::draw_mnist_digit(const std::vector<float> &data) {
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

std::vector<float>
TrainModel::get_mnist_image(linalg::MatrixView<const float> imgs,
                            std::size_t idx) {
  if (idx >= imgs.rows())
    throw std::logic_error("Show prediction: idx out of range");

  const auto D = imgs.cols();
  std::vector<float> out(D);
  for (std::size_t j = 0; j < D; j++)
    out[j] = std::clamp(imgs(idx, j), 0.0f, 1.0f);
  return out;
}

} // namespace Logos::NeuralNet
