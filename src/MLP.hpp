#pragma once

#include "Layers/Linear.hpp"
#include "Layers/ReLU.hpp"
#include "Math/ArenaMatrix.hpp"

namespace Logos::NeuralNet {
static constexpr std::uint32_t INPUT_LAYER = 784, HIDDEN = 256,
                               OUTPUT_LAYER = 10, BATCH_SIZE = 64, EPOCHS = 10;
static constexpr float LEARNING_RATE = 0.05f, LEARNING_RATE_DECAY = 0.95f;

class MLP_Hardcoded {
public:
  MLP_Hardcoded() = delete;
  MLP_Hardcoded(Memory::Arena &m_Arena, std::size_t in_dim,
                std::size_t hidden_dim, std::size_t num_classes,
                std::mt19937 &rng);

  float TrainStep(linalg::MatrixView<const float> X,
                  const std::vector<uint8_t> &labels, float learning_rate);

  void Forward(linalg::MatrixView<const float> X, linalg::MatrixView<float> out,
               bool cache = true);
  void Backward();
  void GradientDescentStep(float learning_rate);

  float Accuracy(linalg::MatrixView<const float> X,
                 const std::vector<uint8_t> &labels);

private:
  Memory::Arena *m_Arena;
  Linear<float> fc1, fc2;
  ReLU<float> relu;

  linalg::ArenaMatrix<float> A, H, logits, probs, dA, dH, dLogits, dX;
};
}; // namespace Logos::NeuralNet
