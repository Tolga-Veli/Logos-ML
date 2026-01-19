#pragma once

#include <random>
#include "Linalg.hpp"

namespace NeuralNet {
class ILayer {
public:
  ILayer() = default;
  virtual ~ILayer() = default;

  virtual void Forward(const linalg::Matrix &in, linalg::Matrix &out) = 0;
  virtual void Backward(const linalg::Matrix &dY, linalg::Matrix &dX) = 0;

  virtual void ZeroGrads() = 0;
  virtual void GradientDescentStep(float learning_rate) = 0;
};

class Linear : public ILayer {
public:
  Linear() = default;
  Linear(size_t in, size_t out, std::mt19937 &rng)
      : m_Weights(in, out), m_Bias(out, 0.0f), m_GradWeights(in, out),
        m_GradBias(out, 0.0f) {

    std::normal_distribution<float> nd(0.0f, std::sqrt(2.0f / float(in)));
    auto X = m_Weights.data();
    for (uint64_t i = 0; i < in; i++)
      for (uint64_t j = 0; j < out; j++)
        X[i * out + j] = nd(rng);
  }
  ~Linear() = default;

  void Forward(const linalg::Matrix &X, linalg::Matrix &H) override {
    if (X.GetCols() != m_Weights.GetRows())
      throw std::logic_error("Wrong input");

    m_LastX = &X;
    linalg::matmul(X, m_Weights, H);
    linalg::add_rowwise_bias(H, m_Bias);
  }

  void Backward(const linalg::Matrix &dA, linalg::Matrix &dX) override {

    if (!m_LastX)
      throw std::runtime_error("Somethinh went wrong");

    if (m_LastX->GetRows() != dA.GetRows() ||
        dA.GetCols() != m_Weights.GetCols() ||
        m_LastX->GetCols() != m_Weights.GetRows())
      throw std::logic_error("Wrong input");

    linalg::matmul_transposeA(*m_LastX, dA, m_GradWeights);
    linalg::sum_rows(dA, m_GradBias);
    linalg::matmul_transposeB(dA, m_Weights, dX);
  }

  void GradientDescentStep(float learning_rate) override {
    const auto N = m_Weights.GetRows(), M = m_Weights.GetCols();

    const auto dX_ptr = m_GradWeights.data();
    auto X = m_Weights.data();
    for (uint64_t i = 0; i < N; i++)
      for (uint64_t j = 0; j < M; j++)
        X[i * M + j] -= learning_rate * dX_ptr[i * M + j];

    for (uint64_t i = 0; i < m_Bias.size(); i++)
      m_Bias[i] -= learning_rate * m_GradBias[i];
  }

  void ZeroGrads() override {
    m_GradWeights.Fill(0.0f);
    std::fill(m_GradBias.begin(), m_GradBias.end(), 0.0f);
  }

private:
  linalg::Matrix m_Weights, m_GradWeights;
  const linalg::Matrix *m_LastX = nullptr;
  std::vector<float> m_Bias, m_GradBias;
};

class ReLU : public ILayer {
public:
  ReLU() = default;
  ~ReLU() = default;
  void Forward(const linalg::Matrix &X, linalg::Matrix &H) override {
    m_LastX = &X;
    const auto N = X.GetRows(), M = X.GetCols();
    if (H.GetRows() != N || H.GetCols() != M)
      H = linalg::Matrix(N, M);

    const auto Y = X.data();
    auto Z = H.data();
    for (size_t i = 0; i < N; i++)
      for (size_t j = 0; j < M; j++)
        Z[i * M + j] = (Y[i * M + j] > 0.0f) ? Y[i * M + j] : 0.0f;
  }

  void Backward(const linalg::Matrix &dH, linalg::Matrix &dA) override {
    if (!m_LastX)
      throw std::runtime_error("Boom");

    const auto N = m_LastX->GetRows(), M = m_LastX->GetCols();
    if (N != dH.GetRows() || M != dH.GetCols())
      throw std::logic_error("ReLU::backward shape mismatch");

    if (dA.GetRows() != N || dA.GetCols() != M)
      dA = linalg::Matrix(N, M);

    const auto X = m_LastX->data(), Y = dH.data();
    auto Z = dA.data();
    for (size_t i = 0; i < N; i++)
      for (size_t j = 0; j < M; j++)
        Z[i * M + j] = (X[i * M + j] > 0.0f) ? Y[i * M + j] : 0.0f;
  }

  void GradientDescentStep(float learning_rate) override {}

  void ZeroGrads() override {}

private:
  const linalg::Matrix *m_LastX = nullptr;
};

} // namespace NeuralNet