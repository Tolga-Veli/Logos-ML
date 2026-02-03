#include "MLP.hpp"

#include "Math/Functions.hpp"

namespace Logos::NeuralNet {
MLP_Hardcoded::MLP_Hardcoded(Memory::Arena &arena, std::size_t in_dim,
                             std::size_t hidden_dim, std::size_t num_classes,
                             std::mt19937 &rng)
    : m_Arena(&arena), fc1(arena, in_dim, hidden_dim, rng),
      fc2(arena, hidden_dim, num_classes, rng),
      A(arena, BATCH_SIZE, hidden_dim), H(arena, BATCH_SIZE, hidden_dim),
      logits(arena, BATCH_SIZE, num_classes),
      probs(arena, BATCH_SIZE, num_classes), dA(arena, BATCH_SIZE, hidden_dim),
      dH(arena, BATCH_SIZE, hidden_dim),
      dLogits(arena, BATCH_SIZE, num_classes), dX(arena, BATCH_SIZE, in_dim) {}

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

  Memory::ScratchArena scratchArena(*m_Arena);
  linalg::ScratchArenaMatrix<float> probs(scratchArena, logits.rows(),
                                          logits.cols());
  Softmax<float>(logits.cview(), probs.view());
  const float loss = CrossEntropy<float>(probs.cview(), labels, dLogits.view());

  Backward();
  GradientDescentStep(learning_rate);
  return loss;
}

void MLP_Hardcoded::Forward(linalg::MatrixView<const float> X,
                            linalg::MatrixView<float> out, bool cache) {
  if (X.rows() != BATCH_SIZE)
    throw std::logic_error("Forward: X.rows() must be BATCH_SIZE");
  if (out.rows() != BATCH_SIZE)
    throw std::logic_error("Forward: out.rows() must be BATCH_SIZE");

  fc1.Forward(X, A.view(), cache);
  relu.Forward(A.cview(), H.view(), cache);
  fc2.Forward(H.cview(), out, cache);
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
} // namespace Logos::NeuralNet
