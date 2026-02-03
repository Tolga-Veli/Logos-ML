#pragma once

#include <stdexcept>
#include <vector>

#include "Layer.hpp"
#include "Math/MatrixView.hpp"

namespace Logos::NeuralNet {
template <class T> class ReLU : public ILayer<T> {
public:
  ReLU() = default;

  void Forward(linalg::MatrixView<const T> in, linalg::MatrixView<T> out,
               bool) override {
    const auto N = in.rows(), M = in.cols();
    if (out.rows() != N || out.cols() != M)
      throw std::logic_error("ReLU::Forward: output shape mismatch");

    m_Rows = N, m_Cols = M;
    m_Mask.resize(N * M);
    for (std::size_t i = 0; i < N; i++)
      for (std::size_t j = 0; j < M; j++) {
        const bool fl = (in(i, j) > T{0});
        m_Mask[i * M + j] = fl;
        out(i, j) = fl ? in(i, j) : T{0};
      }
  }

  void Backward(linalg::MatrixView<const T> dH,
                linalg::MatrixView<T> dX) override {
    if (m_Mask.empty())
      throw std::runtime_error("ReLU::Backward: called before Forward");
    if (dH.rows() != m_Rows || dH.cols() != m_Cols)
      throw std::logic_error("ReLU::Backward: input shape mismatch");
    if (dX.rows() != m_Rows || dX.cols() != m_Cols)
      throw std::logic_error("ReLU::Backward: output shape mismatch");

    for (std::size_t i = 0; i < m_Rows; i++)
      for (std::size_t j = 0; j < m_Cols; j++)
        dX(i, j) = m_Mask[i * m_Cols + j] ? dH(i, j) : T{0};
  }

  void ZeroGrads() override {
    m_Rows = m_Cols = 0;
    m_Mask.clear();
  }
  void GradientDescentStep(float) override {}

private:
  std::size_t m_Rows = 0, m_Cols = 0;
  std::vector<std::uint8_t> m_Mask;
};
} // namespace Logos::NeuralNet
