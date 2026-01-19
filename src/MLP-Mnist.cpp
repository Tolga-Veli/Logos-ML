#include <iostream>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cassert>
#include <random>
#include <limits>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "NeuralNetwork.hpp"

const uint32_t hidden = 256, batch_size = 64, epochs = 10;
float lr = 0.05f; // 0.01

struct MnistImages {
  MnistImages() = default;
  uint32_t num, rows, cols;
  std::vector<uint8_t> bytes;
};

uint32_t read_u32_be(std::ifstream &in) {
  uint8_t buff[4];
  in.read(reinterpret_cast<char *>(buff), 4);
  if (!in)
    throw std::logic_error("Failed to read u32");
  return (uint32_t(buff[0]) << 24) | (uint32_t(buff[1]) << 16) |
         (uint32_t(buff[2]) << 8) | uint32_t(buff[3]);
}

MnistImages load_mnist_images(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::logic_error("Cannot open: " + path);

  const auto magic = read_u32_be(in), num = read_u32_be(in),
             rows = read_u32_be(in), cols = read_u32_be(in);
  if (magic != 2051)
    throw std::logic_error("Bad MNIST image magic in: " + path);
  if (rows == 0 || cols == 0)
    throw std::logic_error("Bad MNIST image dims in: " + path);

  MnistImages out;
  out.num = num;
  out.rows = rows;
  out.cols = cols;
  out.bytes.resize(rows * cols * num);

  in.read(reinterpret_cast<char *>(out.bytes.data()),
          std::streamsize(out.bytes.size()));
  if (!in)
    throw std::logic_error("Failed reading image bytes: " + path);

  return out;
}

std::vector<uint8_t> load_mnist_labels(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::logic_error("Cannot open: " + path);

  const auto magic = read_u32_be(in), num = read_u32_be(in);

  if (magic != 2049)
    throw std::logic_error("Bad MNIST label magic in: " + path);

  std::vector<uint8_t> labels(num);
  in.read(reinterpret_cast<char *>(labels.data()),
          std::streamsize(labels.size()));
  if (!in)
    throw std::logic_error("Failed reading label bytes: " + path);
  return labels;
}

void make_batch(const MnistImages &imgs, const std::vector<uint8_t> &labels,
                const std::vector<uint32_t> &indices, uint32_t start,
                uint32_t batch_size, linalg::Matrix &Xb,
                std::vector<uint8_t> &yb) {
  const auto D = imgs.rows * imgs.cols, N = (uint32_t)indices.size(),
             end = std::min(start + batch_size, N), B = end - start;

  if (B == 0)
    throw std::logic_error("make_batch: empty batch");
  if (Xb.GetRows() != B || Xb.GetCols() != D)
    Xb = linalg::Matrix(B, D);

  yb.resize(B);
  const float inv255 = 1.0f / 255.0f;

  auto X_ptr = Xb.data();
  for (uint32_t i = 0; i < B; i++) {
    const uint32_t idx = indices[start + i];
    yb[i] = labels[idx];

    const std::size_t base = std::size_t(idx) * D;
    for (uint32_t j = 0; j < D; j++)
      X_ptr[i * D + j] = float(imgs.bytes[base + j]) * inv255;
  }
}

void draw_mnist_digit(std::vector<uint8_t> &data) {
  for (uint32_t y = 0; y < 28; y++) {
    for (uint32_t x = 0; x < 28; x++) {
      float num = data[x + y * 28] / 255.0f;
      uint32_t col = 232 + (uint32_t)(num * 23);
      printf("\x1b[48;5;%dm  ", col);
    }
    printf("\n");
  }
  printf("\x1b[0m");
}

std::vector<uint8_t> get_mnist_image(const MnistImages &imgs, uint32_t idx) {
  const uint32_t D = imgs.rows * imgs.cols;
  std::vector<uint8_t> out(D);
  const size_t base = size_t(idx) * D;
  std::copy(imgs.bytes.begin() + base, imgs.bytes.begin() + base + D,
            out.begin());
  return out;
}

void show_prediction(NeuralNet::MLP_Hardcoded &model, const MnistImages &imgs,
                     const std::vector<uint8_t> &labels, uint32_t idx) {

  const uint32_t D = imgs.rows * imgs.cols;
  auto img = get_mnist_image(imgs, idx);
  draw_mnist_digit(img);

  linalg::Matrix X(1, D);
  const float inv255 = 1.0f / 255.0f;
  for (uint32_t j = 0; j < D; j++)
    X.at(0, j) = float(img[j]) * inv255;

  linalg::Matrix logits;
  model.Forward(X, logits);

  uint32_t pred = NeuralNet::ArgmaxRow(logits, 0);

  std::cout << "Prediction: " << pred << " | Ground truth: " << int(labels[idx])
            << '\n'
            << '\n';
}

int main(int argc, char *argv[]) {
  const std::string root = (argc >= 2) ? argv[1] : "data/mnist";
  const std::string train_images_path = root + "/train-images-idx3-ubyte",
                    train_labels_path = root + "/train-labels-idx1-ubyte",
                    test_images_path = root + "/t10k-images-idx3-ubyte",
                    test_labels_path = root + "/t10k-labels-idx1-ubyte";

  std::cout << "Loading MNIST from: " << root << '\n';

  MnistImages train_imgs = load_mnist_images(train_images_path);
  std::vector<uint8_t> train_labels = load_mnist_labels(train_labels_path);

  MnistImages test_imgs = load_mnist_images(test_images_path);
  std::vector<uint8_t> test_labels = load_mnist_labels(test_labels_path);

  if (train_imgs.num != train_labels.size())
    throw std::logic_error("Train images/labels count mismatch");
  if (test_imgs.num != test_labels.size())
    throw std::logic_error("Test images/labels count mismatch");
  if (train_imgs.rows != 28 || train_imgs.cols != 28)
    std::cout << "Warning: train image dims are " << train_imgs.rows << "x"
              << train_imgs.cols << "\n";
  if (test_imgs.rows != train_imgs.rows || test_imgs.cols != train_imgs.cols)
    throw std::logic_error("Train/test image dims mismatch");

  const uint32_t D = train_imgs.rows * train_imgs.cols, C = 10;

  std::mt19937 rng(123);
  // NeuralNet::MLP2 model{std::vector<uint64_t>{D, hidden, hidden, C}, rng};
  NeuralNet::MLP_Hardcoded model{D, hidden, C, rng};

  std::vector<uint32_t> order(train_imgs.num);
  std::iota(order.begin(), order.end(), 0);

  linalg::Matrix Xb;
  std::vector<uint8_t> yb;

  std::cout << "Train: N=" << train_imgs.num << ", D=" << D
            << " | Test: N=" << test_imgs.num << '\n';

  for (uint32_t ep = 1; ep <= epochs; ep++) {
    std::shuffle(order.begin(), order.end(), rng);

    double loss_acc = 0.0f;
    uint32_t steps = 0;

    for (uint32_t start = 0; start < train_imgs.num; start += batch_size) {
      make_batch(train_imgs, train_labels, order, start, batch_size, Xb, yb);
      float loss = model.TrainStep(Xb, yb, lr);

      loss_acc += loss;
      steps++;

      /*if (steps % 300 == 0) {
        uint32_t sample_idx = order[start];
        show_prediction(model, train_imgs, train_labels, sample_idx);
      }*/
    }

    uint64_t correct = 0, total = 0;

    linalg::Matrix Xt;
    std::vector<uint8_t> yt;
    linalg::Matrix logits;

    for (uint32_t start = 0; start < test_imgs.num; start += batch_size) {
      std::vector<uint32_t> test_idx;
      const uint32_t end = std::min(start + batch_size, test_imgs.num);
      test_idx.reserve(end - start);
      for (uint32_t i = start; i < end; i++)
        test_idx.push_back(i);

      make_batch(test_imgs, test_labels, test_idx, 0, (uint32_t)test_idx.size(),
                 Xt, yt);

      model.Forward(Xt, logits);

      for (uint64_t i = 0; i < logits.GetRows(); i++) {
        uint64_t pred = NeuralNet::ArgmaxRow(logits, i);
        if (pred == yt[i])
          correct++;
        total++;
      }
    }

    const float test_acc = (total == 0) ? 0.0f : float(correct) / float(total);

    std::cout << "Epoch " << ep << " done | lr=" << lr
              << " mean_loss=" << float(loss_acc / steps)
              << " test_acc=" << test_acc << '\n';

    lr *= 0.95f;
  }
}