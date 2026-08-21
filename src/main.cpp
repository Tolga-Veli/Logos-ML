#include "Core/DType.hpp"
#include "Core/Shape.hpp"
#include "Core/Tensor.hpp"
#include "Data/DataLoader.hpp"
#include "Memory/MemoryStats.hpp"
#include "Modules/Linear.hpp"
#include "Modules/ReLU.hpp"
#include "Modules/Sequential.hpp"
#include "Ops/Loss.hpp"
#include "Optimizer/SGD.hpp"

#include <chrono>
#include <cmath>
#include <print>

// renders a single image from a (N, 784) tensor to the terminal
inline void render_image(const ml::core::Tensor &images, int index, int label,
                         int predicted = -1) {

  static const char *shades[] = {" ", "░", "▒", "▓", "█"};
  const float *img = images.data<float>() + index * 784;

  std::println();
  for (int r = 0; r < 28; ++r) {
    for (int c = 0; c < 28; ++c) {
      float px = img[r * 28 + c];
      int shade = static_cast<int>(px * 4.99f); // 0-4
      std::print("{}{}", shades[shade],
                 shades[shade]); // doubled for aspect ratio
    }
    std::println();
  }

  if (predicted == -1)
    std::println("Label: {}", label);
  else
    std::println("Label: {}  Predicted: {}  {}", label, predicted,
                 predicted == label ? "✓" : "✗");
}

std::pair<float, float> eval(ml::core::Sequential &model,
                             ml::optim::SGD<float> &optimizer,
                             ml::data::DataLoader &loader, bool test) {

  loader.reset();
  float total_loss = 0.0f;
  int total_batches = 0, correct = 0;

  ml::data::Batch batch;
  if (!loader.next(batch))
    return {0.0f, 0.0f};

  ml::core::Tensor logits(model.output_shape(batch.images.shape())),
      probs(logits.shape(), logits.dtype()), grad(probs.shape(), probs.dtype()),
      X, loss(ml::core::Shape{});

  while (loader.next(batch)) {
    model.forward(batch.images, logits);
    ml::ops::cross_entropy(logits, batch.labels, probs, loss);

    float loss_value = *loss.data<float>();

    int batch_size = batch.images.shape()[0];
    total_loss += loss_value * static_cast<float>(batch_size);
    total_batches += batch_size;

    const float *prob_ptr = probs.data<float>();
    const int *labels = batch.labels.data<int>();
    for (int i = 0; i < batch.images.shape()[0]; i++) {
      const float *ptr = prob_ptr + i * 10;

      int pred = 0;
      float best = ptr[0];

      for (int j = 1; j < 10; j++)
        if (ptr[j] > best) {
          best = ptr[j];
          pred = j;
        }

      if (pred == labels[i])
        ++correct;

      if (test) {
        render_image(batch.images, i, labels[i], pred);

        std::println("\nProbabilities:");
        for (int j = 0; j < 10; j++)
          std::println("{} : {:.2f}%%", j, ptr[j] * 100.0f);

        std::print("\nPress enter for next image...");
        getchar();
      }
    }

    optimizer.zero_grad();
    ml::ops::cross_entropy_backward(probs, batch.labels, grad);
    model.backward(grad, X);
    optimizer.step();
  }

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

  ml::data::DataLoader train_loader(std::move(train_images),
                                    std::move(train_labels), BATCH_SIZE, true);
  ml::data::DataLoader test_loader(std::move(test_images),
                                   std::move(test_labels), BATCH_SIZE, false);

  ml::core::Sequential model;
  model.add<ml::core::Linear>(784, 128, ml::core::DType::Float32);
  model.add<ml::core::ReLU>();
  model.add<ml::core::Linear>(128, 64, ml::core::DType::Float32);
  model.add<ml::core::ReLU>();
  model.add<ml::core::Linear>(64, 10, ml::core::DType::Float32);

  ml::optim::SGD<float> optimizer(model.parameters(), LEARNING_RATE, MOMENTUM,
                                  WEIGHT_DECAY);
  ml::data::Batch batch;
  for (int epoch = 1; epoch <= EPOCHS; epoch++) {
    auto start_alloc_cnt = ml::memory::get_stats().allocations;
    auto epochStart = std::chrono::steady_clock::now();

    auto [loss, acc] = eval(model, optimizer, train_loader, false);

    auto epochMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - epochStart)
                       .count();
    auto end_alloc_cnt = ml::memory::get_stats().allocations;

    std::println("Epoch {:2} | Loss {:.4f} | Accuracy {:.4f}%", epoch, loss,
                 acc * 100.0f);
    std::println("Allocations: {:2} | Time: {:2}ms\n",
                 end_alloc_cnt - start_alloc_cnt, epochMs);
  }

  eval(model, optimizer, test_loader, true);
}
