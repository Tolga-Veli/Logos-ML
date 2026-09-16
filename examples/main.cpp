#include "Core/DataLoader.hpp"
#include "Core/Shape.hpp"
#include "Core/Tensor.hpp"
#include "Modules/Linear.hpp"
#include "Modules/ReLU.hpp"
#include "Modules/Sequential.hpp"
#include "Ops/Loss.hpp"
#include "Optimizers/SGD.hpp"

#include <algorithm>
#include <cmath>
#include <print>

// renders a single image from a (N, 784) tensor to the terminal
inline void render_image(const ml::core::Tensor &images, int index, int label, int pred = -1) {
  static const char *shades[] = {" ", "░", "▒", "▓", "█"};

  std::println();
  for (int r = 0; r < 28; r++) {
    for (int c = 0; c < 28; c++) {
      const float px = images.at<float>(index, r * 28 + c);
      const int shade = std::clamp(static_cast<int>(px * 4.99f), 0, 4);

      std::print("{}{}", shades[shade], shades[shade]);
    }
    std::println();
  }

  if (pred == -1)
    std::println("Label: {}", label);
  else
    std::println("Label: {}  Predicted: {}  {}", label, pred, pred == label ? "✓" : "✗");
}

std::pair<float, float> eval(ml::core::Sequential &model, ml::optim::SGD<float> &optimizer,
                             ml::core::DataLoader &loader, bool test) {

  loader.reset();
  float total_loss = 0.0f;
  int total_batches = 0, correct = 0;

  ml::core::Batch batch;
  ml::core::Tensor logits, probs, grad, X, loss;
  while (loader.next(batch)) {
    if (total_batches == 0) {
      logits = ml::core::Tensor(model.output_shape(batch.images.shape()));
      probs = ml::core::Tensor(logits.shape(), logits.dtype());
      grad = ml::core::Tensor(probs.shape(), probs.dtype());
      loss = ml::core::Tensor(ml::core::Shape{});
    }

    model.forward(batch.images, logits);
    ml::ops::cross_entropy(logits, batch.labels, probs, loss);

    const float loss_value = loss.at<float>();

    int batch_size = batch.images.shape()[0];
    total_loss += loss_value * batch_size;
    total_batches += batch_size;

    const int classes = probs.shape()[1];
    for (int i = 0; i < batch_size; i++) {
      int pred = 0;
      float best = probs.at<float>(i, 0);

      for (int j = 1; j < classes; j++) {
        const float prob = probs.at<float>(i, j);

        if (prob > best) {
          best = prob;
          pred = j;
        }
      }

      const int label = batch.labels.at<int>(i);
      if (pred == label)
        ++correct;

      if (test) {
        render_image(batch.images, i, label, pred);

        std::println("\nProbabilities:");
        for (int j = 0; j < classes; j++)
          std::println("{} : {:.2f}%%", j, probs.at<float>(i, j) * 100.0f);

        std::print("\nPress enter for next image...");
        getchar();
      }
    }

    if (!test) {
      optimizer.zero_grad();
      ml::ops::cross_entropy_backward(probs, batch.labels, grad);
      model.backward(grad, X);
      optimizer.step();
    }
  }

  if (total_batches == 0)
    return {0.0f, 0.0f};

  return {total_loss / static_cast<float>(total_batches),
          static_cast<float>(correct) / static_cast<float>(total_batches)};
}

int main() {
  auto train_images = ml::core::load_binary<float>("data/train_images.bin", {60'000, 784});
  auto train_labels = ml::core::load_binary<int>("data/train_labels.bin", {60'000});

  auto test_images = ml::core::load_binary<float>("data/test_images.bin", {10'000, 784});
  auto test_labels = ml::core::load_binary<int>("data/test_labels.bin", {10'000});

  constexpr std::size_t BATCH_SIZE = 32, EPOCHS = 10;
  constexpr float LEARNING_RATE = 0.01f, MOMENTUM = 0.0f, WEIGHT_DECAY = 0.0f;

  ml::core::DataLoader train_loader(std::move(train_images), std::move(train_labels), BATCH_SIZE, true);
  ml::core::DataLoader test_loader(std::move(test_images), std::move(test_labels), BATCH_SIZE, false);

  ml::core::Sequential model;
  model.add<ml::core::Linear>(784, 256);
  model.add<ml::core::ReLU>();
  model.add<ml::core::Linear>(256, 128);
  model.add<ml::core::ReLU>();
  model.add<ml::core::Linear>(128, 10);

  ml::optim::SGD<float> optimizer(model.parameters(), LEARNING_RATE, MOMENTUM, WEIGHT_DECAY);
  ml::core::Batch batch;
  for (std::size_t epoch = 1; epoch <= EPOCHS; epoch++) {
    auto [loss, acc] = eval(model, optimizer, train_loader, false);
    LOG_INFO("Epoch {:2} | Loss {:.4f} | Accuracy {:.4f}%", epoch, loss, acc * 100.0f);
  }

  eval(model, optimizer, test_loader, true);
}
