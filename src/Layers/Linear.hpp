#pragma once

#include <random>

#include "Kernels/ArenaMatrixOperations.hpp"
#include "Layer.hpp"
#include "Math/ArenaMatrix.hpp"
#include "Math/MatrixView.hpp"

namespace Logos::NeuralNet {
template <class T> class Linear : public ILayer<T> {
public:
  Linear() = default;
  Linear(Memory::Arena &m_Arena, std::size_t in, std::size_t out,
         std::mt19937 &rng)
      : m_Weights(m_Arena, in, out), m_GradWeights(m_Arena, in, out),
        m_Bias(out), m_GradBias(out), m_LastX() {

    const T upper_lim = std::sqrt(T(2) / static_cast<T>(in));
    std::normal_distribution<T> nd(T(0), upper_lim);

    auto X = m_Weights.data();
    for (std::size_t i = 0; i < in; i++)
      for (std::size_t j = 0; j < out; j++)
        X[i * out + j] = nd(rng);

    ZeroGrads();
  }
  ~Linear() = default;

  void Forward(linalg::MatrixView<const T> in, linalg::MatrixView<T> out,
               bool cache = true) override {
    if (in.cols() != m_Weights.rows())
      throw std::logic_error("Wrong input");

    if (cache)
      m_LastX = in;
    linalg::matmul(in, m_Weights.cview(), out);
    linalg::add_rowwise_bias(m_Bias, out);
  }

  void Backward(linalg::MatrixView<const T> prev,
                linalg::MatrixView<T> curr) override {
    if (m_LastX.rows() != prev.rows() || prev.cols() != m_Weights.cols() ||
        m_LastX.cols() != m_Weights.rows())
      throw std::logic_error("Wrong input");

    linalg::matmul_transposeA(m_LastX, prev, m_GradWeights.view());
    linalg::sum_rows(prev, m_GradBias);
    linalg::matmul_transposeB(prev, m_Weights.cview(), curr);
  }

  void GradientDescentStep(float learning_rate) override {
    const auto N = m_Weights.rows(), M = m_Weights.cols();
    const auto dWeights = m_GradWeights.data();
    auto weights = m_Weights.data();
    for (std::size_t i = 0; i < N; i++)
      for (std::size_t j = 0; j < M; j++)
        weights[i * M + j] -= learning_rate * dWeights[i * M + j];

    for (std::size_t i = 0; i < m_Bias.size(); i++)
      m_Bias[i] -= learning_rate * m_GradBias[i];
  }

  void ZeroGrads() override {
    std::fill(m_GradBias.begin(), m_GradBias.end(), 0);
    m_GradWeights.fill_zeroes();
  }

private:
  linalg::ArenaMatrix<T> m_Weights, m_GradWeights;
  std::vector<T> m_Bias, m_GradBias;
  linalg::MatrixView<const T> m_LastX;
};
} // namespace Logos::NeuralNet
