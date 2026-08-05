#include "Data/DataLoader.hpp"
#include "Math/loss.hpp"
#include "Modules/Linear.hpp"
#include "Modules/ReLU.hpp"
#include "Modules/Sequential.hpp"
#include "Optimizer/SGD.hpp"

// renders a single image from a (N, 784) tensor to the terminal
inline void render_image(const ml::core::Tensor<float> &images, int index,
                         int label, int predicted = -1) {

  static const char *shades[] = {" ", "░", "▒", "▓", "█"};
  const float *img = images.data() + index * 784;

  printf("\n");
  for (int r = 0; r < 28; ++r) {
    for (int c = 0; c < 28; ++c) {
      float px = img[r * 28 + c];
      int shade = static_cast<int>(px * 4.99f);     // 0-4
      printf("%s%s", shades[shade], shades[shade]); // doubled for aspect ratio
    }
    printf("\n");
  }

  if (predicted == -1)
    printf("Label: %d\n", label);
  else
    printf("Label: %d  Predicted: %d  %s\n", label, predicted,
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

  ml::core::Sequential<float> model;
  model.add<ml::core::Linear<float>>(784, 128);
  model.add<ml::core::ReLU<float>>();
  model.add<ml::core::Linear<float>>(128, 64);
  model.add<ml::core::ReLU<float>>();
  model.add<ml::core::Linear<float>>(64, 10);

  ml::optim::SGD<float> optimizer(model.parameters(), LEARNING_RATE, MOMENTUM,
                                  WEIGHT_DECAY);

  for (int epoch = 1; epoch <= EPOCHS; epoch++) {
    train_loader.reset();
    cnt = 0;

    float total_loss = 0.0f;
    int batches = 0;
    ml::data::Batch batch;

    while (train_loader.next(batch)) {
      auto logits = model.forward(batch.images);
      auto [loss, probs] = ml::ops::cross_entropy(logits, batch.labels);

      batches++;
      total_loss += loss;

      optimizer.zero_grad();

      auto grad = ml::ops::cross_entropy_backward(probs, batch.labels);

      model.backward(grad);
      optimizer.step();
    }

    printf("Epoch %2d loss %.4f\n", epoch, total_loss / batches);
    printf("Allocations: %d\n", cnt);
  }
}
