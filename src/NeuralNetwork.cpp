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

#include "Functions.hpp"
#include "NeuralNetwork.hpp"

namespace Logos::NeuralNet {

MLP_Hardcoded::MLP_Hardcoded(std::size_t in_dim, std::size_t hidden_dim,
                             std::size_t num_classes, std::mt19937 &rng)
    : fc1(in_dim, hidden_dim, rng), fc2(hidden_dim, num_classes, rng) {}

double MLP_Hardcoded::TrainStep(const Matrix &X,
                                const std::vector<uint8_t> &labels,
                                double learning_rate) {
  const auto N = X.rows(), M = X.cols();
  if (N == 0 || M == 0)
    throw std::logic_error("TrainStep: empty input matrix");
  if (labels.size() != N)
    throw std::logic_error("TrainStep: labels size mismatch");

  fc1.Forward(X, A1);
  relu.Forward(A1, H1);
  fc2.Forward(H1, logits);

  Matrix probs;
  Softmax<float>(logits, probs);
  const float loss = CrossEntropy<float>(probs, labels, dLogits);

  fc2.Backward(dLogits, dH1);
  relu.Backward(dH1, dA1);
  fc1.Backward(dA1, dX);

  fc1.GradientDescentStep(learning_rate);
  fc2.GradientDescentStep(learning_rate);

  fc1.ZeroGrads();
  fc2.ZeroGrads();

  return loss;
}

void MLP_Hardcoded::Forward(const Matrix &X, Matrix &out) {
  fc1.Forward(X, A1);
  relu.Forward(A1, H1);
  fc2.Forward(H1, out);
}

double MLP_Hardcoded::Accuracy(const Matrix &X,
                               const std::vector<uint8_t> &labels) {
  Forward(X, logits);
  const auto N = logits.rows();
  if (N != labels.size() || N == 0)
    throw std::logic_error("MLP_Hardcoded::Accuracy wrong Matrix size");

  std::uint32_t cnt = 0;
  for (std::size_t i = 0; i < N; i++) {
    std::size_t pred = ArgmaxRow<float>(logits, i);
    if (pred == labels[i])
      cnt++;
  }

  return static_cast<double>(cnt) / N;
}

TrainModel::TrainModel()
    : m_RNG(123), m_LearningRate(LEARNING_RATE),
      m_TrainImgs(load_images_mat("data/train_images.mat", 60000, 28, 28)),
      m_TrainLabels(load_labels_mat("data/train_labels.mat", 60000)),
      m_TestImgs(load_images_mat("data/test_images.mat", 10000, 28, 28)),
      m_TestLabels(load_labels_mat("data/test_labels.mat", 10000)),
      m_Model(INPUT_LAYER, HIDDEN, OUTPUT_LAYER, m_RNG) {

  m_Order.resize(m_TrainImgs.rows());
  std::iota(m_Order.begin(), m_Order.end(), 0);

  std::cout << "Train: N=" << m_TrainImgs.rows()
            << " | Test: N=" << m_TestImgs.rows() << '\n';
}

void TrainModel::run() {
  Matrix Xb;
  std::vector<uint8_t> yb;

  for (std::uint32_t ep = 1; ep <= EPOCHS; ep++) {
    std::shuffle(m_Order.begin(), m_Order.end(), m_RNG);

    double loss_acc = 0.0;
    std::size_t steps = 0;

    for (std::size_t start = 0; start < m_Order.size(); start += BATCH_SIZE) {
      make_batch(m_TrainImgs, m_TrainLabels, m_Order, start, BATCH_SIZE, Xb,
                 yb);

      const double loss = m_Model.TrainStep(Xb, yb, m_LearningRate);
      loss_acc += loss;
      steps++;

      if (steps % 500 == 0)
        show_prediction(m_Model, m_TrainImgs, m_TrainLabels, m_Order[start]);
    }

    std::uint32_t correct = 0, total = 0;
    Matrix Xt, logits;
    std::vector<uint8_t> yt;

    for (std::size_t start = 0; start < m_TestImgs.rows();
         start += BATCH_SIZE) {
      const auto end = std::min(start + BATCH_SIZE, m_TestImgs.rows());

      std::vector<std::size_t> test_idx(end - start);
      iota(test_idx.begin(), test_idx.end(), start);

      make_batch(m_TestImgs, m_TestLabels, test_idx, 0, test_idx.size(), Xt,
                 yt);

      m_Model.Forward(Xt, logits);

      for (std::size_t i = 0; i < logits.rows(); i++) {
        const std::size_t pred = Logos::NeuralNet::ArgmaxRow<float>(logits, i);
        if (pred == yt[i])
          correct++;
        total++;
      }
    }

    const double test_acc =
        (total == 0) ? 0.0f : static_cast<float>(correct) / total;
    const double mean_loss = (steps == 0) ? 0.0f : loss_acc / steps;

    std::cout << "Epoch " << ep << " done | lr=" << m_LearningRate
              << " mean_loss=" << mean_loss << " test_acc=" << test_acc << '\n';

    m_LearningRate *= LEARNING_RATE_DECAY;
  }
}

Matrix TrainModel::load_images_mat(std::string path, std::size_t num,
                                   std::size_t rows, std::size_t cols) {
  const auto D = rows * cols, total = num * D;
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Cannot open: " + path);

  Matrix out(num, D);
  in.read(reinterpret_cast<char *>(out.data()),
          static_cast<std::streamsize>(total * sizeof(float)));

  if (!in)
    throw std::runtime_error("Failed reading: " + path);

  return out;
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

void TrainModel::make_batch(const Matrix &imgs,
                            const std::vector<std::uint8_t> &labels,
                            const std::vector<std::size_t> &indices,
                            std::size_t start, std::size_t batch_size,
                            Matrix &Xb, std::vector<std::uint8_t> &yb) {

  const auto D = imgs.cols(), N = indices.size(),
             end = std::min(start + static_cast<std::size_t>(batch_size), N),
             B = end - start;

  if (B == 0)
    throw std::logic_error("make_batch: empty batch");

  if (Xb.rows() != B || Xb.cols() != D)
    Xb = Matrix(B, D);

  yb.resize(B);
  for (std::size_t i = 0; i < B; i++) {
    const auto idx = indices[start + i];
    yb[i] = labels[idx];

    for (std::size_t j = 0; j < D; j++)
      Xb(i, j) = imgs(idx, j);
  }
}

void TrainModel::show_prediction(NeuralNetwork &model, const Matrix &imgs,
                                 const std::vector<uint8_t> &labels,
                                 std::size_t idx) {
  std::vector<float> img = get_mnist_image(imgs, idx);
  draw_mnist_digit(img);

  const auto D = imgs.cols();
  Matrix X(1, D);

  for (std::size_t j = 0; j < D; j++)
    X(0, j) = img[j];

  Matrix logits;
  model.Forward(X, logits);

  std::cout << "\nPrediction: " << Logos::NeuralNet::ArgmaxRow<float>(logits, 0)
            << " | Ground truth: " << static_cast<int>(labels[idx]) << "\n\n";
}

void TrainModel::draw_mnist_digit(const std::vector<float> &data) {
  for (std::size_t x = 0; x < 28; x++) {
    for (std::size_t y = 0; y < 28; y++) {
      const float num = data[x * 28 + y];
      const std::uint32_t col = 232u + static_cast<std::uint32_t>(num * 23.0f);
      std::printf("\x1b[48;5;%dm  ", col);
    }
    std::printf("\n");
  }
  std::printf("\x1b[0m");
}

std::vector<float> TrainModel::get_mnist_image(const Matrix &imgs,
                                               std::size_t idx) {
  const auto D = imgs.cols();
  std::vector<float> out(D);

  for (std::size_t j = 0; j < D; j++)
    out[j] = std::clamp(imgs(idx, j), 0.0f, 1.0f);
  return out;
}

} // namespace Logos::NeuralNet
