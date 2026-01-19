#pragma once

#include <memory>

#include "Linalg.hpp"
#include "Functions.hpp"
#include "Layer.hpp"

namespace NeuralNet {
class MLP_Hardcoded {
public:
  MLP_Hardcoded(uint64_t in_dim, uint64_t hidden_dim, uint64_t num_classes,
                std::mt19937 &rng)
      : fc1(in_dim, hidden_dim, rng), fc2(hidden_dim, num_classes, rng) {}

  float TrainStep(const linalg::Matrix &X, const std::vector<uint8_t> &labels,
                  float learning_rate) {
    const auto N = X.GetRows(), M = X.GetCols();
    if (N == 0 || M == 0)
      throw std::logic_error("TrainStep: empty input matrix");
    if (labels.size() != N)
      throw std::logic_error("TrainStep: labels size mismatch");

    fc1.Forward(X, A1);
    relu.Forward(A1, H1);
    fc2.Forward(H1, logits);

    float loss = SoftmaxCrossEntropyFromLogits(logits, labels, dLogits);

    fc2.Backward(dLogits, dH1);
    relu.Backward(dH1, dA1);
    fc1.Backward(dA1, dX);
    fc1.GradientDescentStep(learning_rate);
    fc2.GradientDescentStep(learning_rate);

    return loss;
  }

  void Forward(const linalg::Matrix &X, linalg::Matrix &out) {
    fc1.Forward(X, A1);
    relu.Forward(A1, H1);
    fc2.Forward(H1, out);
  }

  float Accuracy(const linalg::Matrix &X, const std::vector<uint8_t> &labels) {
    Forward(X, logits);
    const auto N = logits.GetRows();
    if (N != labels.size())
      throw std::logic_error("AccuracyFromLogits: labels size mismatch");

    uint64_t cnt = 0;
    for (uint64_t i = 0; i < N; i++) {
      uint64_t pred = ArgmaxRow(logits, i);
      if (pred == labels[i])
        cnt++;
    }

    return (N == 0) ? 0.0f : static_cast<float>(cnt) / static_cast<float>(N);
  }

private:
  Linear fc1, fc2;
  ReLU relu;
  // std::vector<std::unique_ptr<ILayer>> m_Layers;

  linalg::Matrix A1, H1, logits, probs, dA1, dH1, dLogits, dX;
};

class MLP2 {
public:
  MLP2(const std::vector<uint64_t> &dims, std::mt19937 &rng) {
    if (dims.size() < 2)
      throw std::logic_error("MLP: need at least input and output dims");

    for (size_t i = 0; i + 1 < dims.size(); i++) {
      m_Layers.push_back(std::make_unique<Linear>(dims[i], dims[i + 1], rng));
      if (i + 2 < dims.size())
        m_Layers.push_back(std::make_unique<ReLU>());
    }
  }

  void Forward(const linalg::Matrix &X, linalg::Matrix &out_logits) {
    if (m_Layers.empty())
      throw std::logic_error("Forward: no layers");

    const linalg::Matrix *curr = &X;
    linalg::Matrix *buf0 = &A0;
    linalg::Matrix *buf1 = &A1;
    bool flip = false;

    for (size_t i = 0; i + 1 < m_Layers.size(); i++) {
      linalg::Matrix *nxt = flip ? buf1 : buf0;
      m_Layers[i]->Forward(*curr, *nxt);
      curr = nxt;
      flip = !flip;
    }

    m_Layers.back()->Forward(*curr, out_logits);
  }

  float TrainStep(const linalg::Matrix &X, const std::vector<uint8_t> &labels,
                  float learning_rate) {
    const auto N = X.GetRows(), M = X.GetCols();
    if (N == 0 || M == 0)
      throw std::logic_error("TrainStep: empty input matrix");
    if (labels.size() != N)
      throw std::logic_error("TrainStep: labels size mismatch");
    if (m_Layers.empty())
      throw std::logic_error("TrainStep: no layers");

    Forward(X, m_Logits);

    float loss = SoftmaxCrossEntropyFromLogits(m_Logits, labels, dLogits);

    const linalg::Matrix *dcurr = &dLogits;
    linalg::Matrix *gbuf0 = &G0;
    linalg::Matrix *gbuf1 = &G1;
    bool flip = false;

    for (size_t i = m_Layers.size(); i-- > 0;) {
      linalg::Matrix *dnxt = flip ? gbuf1 : gbuf0;
      m_Layers[i]->Backward(*dcurr, *dnxt);
      dcurr = dnxt;
      flip = !flip;
    }

    for (auto &layer : m_Layers) {
      layer->GradientDescentStep(learning_rate);
      layer->ZeroGrads();
    }

    return loss;
  }

  float Accuracy(const linalg::Matrix &X, const std::vector<uint8_t> &labels) {
    Forward(X, m_Logits);
    const auto N = m_Logits.GetRows();
    if (N != labels.size())
      throw std::logic_error("Accuracy: labels size mismatch");

    uint64_t cnt = 0;
    for (uint64_t i = 0; i < N; i++) {
      uint64_t pred = ArgmaxRow(m_Logits, i);
      if (pred == labels[i])
        cnt++;
    }
    return (N == 0) ? 0.0f : static_cast<float>(cnt) / static_cast<float>(N);
  }

private:
  std::vector<std::unique_ptr<ILayer>> m_Layers;

  linalg::Matrix A0, A1;
  linalg::Matrix G0, G1;
  linalg::Matrix probs, dLogits;

  linalg::Matrix m_Logits;
};

} // namespace NeuralNet
