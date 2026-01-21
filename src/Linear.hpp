#pragma once

#include "Kernels.hpp"
#include "Layer.hpp"
#include <random>

namespace Logos::NeuralNet {
template <class T> class Linear : public ILayer<T> {
public:
  Linear() = default;
  Linear(std::size_t in, std::size_t out, std::mt19937 &rng)
      : m_Weights(in, out), m_GradWeights(in, out), m_Bias(out),
        m_GradBias(out), m_LastX(nullptr), m_HasLastX(false) {

    const T upper_lim = std::sqrt(T(2) / static_cast<T>(in));
    std::normal_distribution<T> nd(T(0), upper_lim);

    auto X = m_Weights.data();
    // row-major
    for (std::size_t i = 0; i < in; i++)
      for (std::size_t j = 0; j < out; j++)
        X[i * out + j] = nd(rng);

    ZeroGrads();
  }
  ~Linear() = default;

  void Forward(const linalg::Matrix<T> &X, linalg::Matrix<T> &H) override {
    if (X.cols() != m_Weights.rows())
      throw std::logic_error("Wrong input");

    m_LastX = &X;
    m_HasLastX = true;

    linalg::matmul<T>(X, m_Weights, H);
    linalg::add_rowwise_bias<T>(m_Bias, H);
  }

  void Backward(const linalg::Matrix<T> &dA, linalg::Matrix<T> &dX) override {
    if (!m_HasLastX)
      throw std::runtime_error("Somethinh went wrong");

    if (m_LastX->rows() != dA.rows() || dA.cols() != m_Weights.cols() ||
        m_LastX->cols() != m_Weights.rows())
      throw std::logic_error("Wrong input");

    linalg::matmul_transposeA<T>(*m_LastX, dA, m_GradWeights);
    linalg::sum_rows<T>(dA, m_GradBias);
    linalg::matmul_transposeB<T>(dA, m_Weights, dX);
  }

  void GradientDescentStep(float learning_rate) override {
    const auto N = m_Weights.rows(), M = m_Weights.cols();
    const auto dX_ptr = m_GradWeights.data();
    auto X = m_Weights.data();
    for (std::size_t i = 0; i < N; i++)
      for (std::size_t j = 0; j < M; j++)
        X[i * M + j] -= learning_rate * dX_ptr[i * M + j];

    for (std::size_t i = 0; i < m_Bias.size(); i++)
      m_Bias[i] -= learning_rate * m_GradBias[i];
  }

  void ZeroGrads() override {
    m_GradWeights.fill_zeroes();
    std::fill(m_GradBias.begin(), m_GradBias.end(), T{0});
  }

private:
  linalg::Matrix<T> m_Weights, m_GradWeights;
  std::vector<T> m_Bias, m_GradBias;

  const linalg::Matrix<T> *m_LastX = nullptr;
  bool m_HasLastX = false;
};
} // namespace Logos::NeuralNet
