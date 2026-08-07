#include "Core/DType.hpp"
#include "Core/Tensor.hpp"
#include "Data/DataLoader.hpp"
#include "Modules/Linear.hpp"
#include "Modules/ReLU.hpp"
#include "Modules/Sequential.hpp"
#include "Ops/Loss.hpp"
#include "Optimizer/SGD.hpp"

#include <chrono>
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

int cnt = 0;

void *operator new(std::size_t size) {
  cnt++;
  if (void *ptr = std::malloc(size))
    return ptr;

  throw std::bad_alloc{};
}

void operator delete(void *ptr) noexcept { std::free(ptr); }

void *operator new[](std::size_t size) {
  cnt++;
  if (void *ptr = std::malloc(size))
    return ptr;

  throw std::bad_alloc{};
}

void operator delete[](void *ptr) noexcept { std::free(ptr); }

float eval(ml::core::Sequential &model, ml::optim::SGD<float> &optimizer,
           ml::data::DataLoader &loader, bool train) {
  loader.reset();
  cnt = 0;

  float total_loss = 0.0f;
  int batches = 0;

  ml::data::Batch batch;
  while (loader.next(batch)) {
    ml::core::Tensor logits;
    model.forward(batch.images, logits);

    ml::core::Tensor probs(logits.shape(), logits.dtype());
    auto loss = ml::ops::cross_entropy<float>(logits, batch.labels, probs);

    if (train) {
      batches++;
      total_loss += loss;

      optimizer.zero_grad();

      ml::core::Tensor grad(probs.shape(), probs.dtype());
      ml::ops::cross_entropy_backward<float>(probs, batch.labels, grad);

      ml::core::Tensor X;
      model.backward(grad, X);
      optimizer.step();
    } else {
      for (int i = 0; i < batch.images.shape()[0]; i++) {
        const float *ptr = probs.data<float>() + i * 10;

        int pred = 0;
        float best = ptr[0];

        for (int j = 1; j < 10; j++)
          if (ptr[j] > best) {
            best = ptr[j];
            pred = j;
          }

        int label = static_cast<int>(batch.labels.data<int>()[i]);
        render_image(batch.images, i, label, pred);

        std::println("\nProbabilities:");
        for (int j = 0; j < 10; j++)
          std::println("{} : {:.2f}%%", j, ptr[j] * 100.0f);

        std::print("\nPress enter for next image...");
        getchar();
      }
    }
  }

  return total_loss / static_cast<float>(batches);
}

int main() {
  auto train_images =
      ml::data::load_binary("data/train_images.bin", {60'000, 784});
  auto train_labels =
      ml::data::load_binary<int>("data/train_labels.bin", {60'000});

  auto test_images =
      ml::data::load_binary("data/test_images.bin", {10'000, 784});
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
    auto epochStart = std::chrono::steady_clock::now();

    auto loss = eval(model, optimizer, train_loader, true);

    auto epochEnd = std::chrono::steady_clock::now();
    auto epochMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                       epochEnd - epochStart)
                       .count();
    std::println("Epoch {} loss {:.4f}", epoch, loss);
    std::println("Allocations: {} | Time: {}ms\n", cnt, epochMs);
  }

  eval(model, optimizer, test_loader, false);
}
