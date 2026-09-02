#include "Core/Shape.hpp"
#include "Core/Tensor.hpp"
#include "Data/DataLoader.hpp"
#include "Modules/Linear.hpp"
#include "Modules/ReLU.hpp"
#include "Modules/Sequential.hpp"
#include "Ops/Loss.hpp"
#include "Optimizer/SGD.hpp"

#include "Debug/ScopedAllocationCounter.hpp"
#include "Debug/ScopedTimer.hpp"

#include <algorithm>
#include <cmath>
#include <print>

// renders a single image from a (N, 784) tensor to the terminal
inline void render_image(const ml::core::Tensor &images, int index, int label,
                         int pred = -1) {

  static const char *shades[] = {" ", "░", "▒", "▓", "█"};
  const float *img = images.data<float>() + index * 784;

  std::println();
  for (int r = 0; r < 28; ++r) {
    for (int c = 0; c < 28; ++c) {
      float px = img[r * 28 + c];
      int shade = std::clamp(static_cast<int>(px * 4.99f), 0, 4);
      std::print("{}{}", shades[shade],
                 shades[shade]); // doubled for aspect ratio
    }
    std::println();
  }

  if (pred == -1)
    std::println("Label: {}", label);
  else
    std::println("Label: {}  Predicted: {}  {}", label, pred,
                 pred == label ? "✓" : "✗");
}

std::pair<float, float> eval(ml::core::Sequential &model,
                             ml::optim::SGD<float> &optimizer,
                             ml::data::DataLoader &loader, bool test) {

  loader.reset();
  float total_loss = 0.0f;
  int total_batches = 0, correct = 0;

  ml::data::Batch batch;
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

    float loss_value = *loss.data<float>();

    int batch_size = batch.images.shape()[0];
    total_loss += loss_value * static_cast<float>(batch_size);
    total_batches += batch_size;

    const float *prob_ptr = probs.data<float>();
    const int *labels = batch.labels.data<int>();
    const int classes = probs.shape()[1];
    for (int i = 0; i < batch.images.shape()[0]; i++) {
      const float *ptr = prob_ptr + i * classes;

      int pred = 0;
      float best = ptr[0];

      for (int j = 1; j < classes; j++)
        if (ptr[j] > best) {
          best = ptr[j];
          pred = j;
        }

      if (pred == labels[i])
        ++correct;

      if (test) {
        render_image(batch.images, i, labels[i], pred);

        std::println("\nProbabilities:");
        for (int j = 0; j < classes; j++)
          std::println("{} : {:.2f}%%", j, ptr[j] * 100.0f);

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
  auto train_images =
      ml::data::load_binary<float>("data/train_images.bin", {60'000, 784});
  auto train_labels =
      ml::data::load_binary<int>("data/train_labels.bin", {60'000});

  auto test_images =
      ml::data::load_binary<float>("data/test_images.bin", {10'000, 784});
  auto test_labels =
      ml::data::load_binary<int>("data/test_labels.bin", {10'000});

  constexpr int BATCH_SIZE = 32, EPOCHS = 10;
  constexpr float LEARNING_RATE = 0.01f, MOMENTUM = 0.0f, WEIGHT_DECAY = 0.0f;
  constexpr bool LOG = false;

  ml::data::DataLoader train_loader(std::move(train_images),
                                    std::move(train_labels), BATCH_SIZE, true);
  ml::data::DataLoader test_loader(std::move(test_images),
                                   std::move(test_labels), BATCH_SIZE, false);

  ml::core::Sequential model;
  model.add<ml::core::Linear>(784, 256);
  model.add<ml::core::ReLU>();
  model.add<ml::core::Linear>(256, 128);
  model.add<ml::core::ReLU>();
  model.add<ml::core::Linear>(128, 10);

  ml::optim::SGD<float> optimizer(model.parameters(), LEARNING_RATE, MOMENTUM,
                                  WEIGHT_DECAY);
  ml::data::Batch batch;
  for (int epoch = 1; epoch <= EPOCHS; epoch++) {
    ml::debug::ScopedAllocationCounter alloc(LOG);
    ml::debug::ScopedTimer timer(LOG);

    auto [loss, acc] = eval(model, optimizer, train_loader, false);

    LOG_INFO("Epoch {:2} | Loss {:.4f} | Accuracy {:.4f}%", epoch, loss,
             acc * 100.0f);
  }

  eval(model, optimizer, test_loader, true);
}
