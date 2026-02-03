#pragma once

#include <stdexcept>
#include <vector>

#include "Layer.hpp"
#include "Math/MatrixView.hpp"

namespace Logos::NeuralNet {
template <class T> class ReLU : public ILayer<T> {
public:
  ReLU() = default;

  void Forward(linalg::MatrixView<const T> in,
               linalg::MatrixView<T> out) override {
    const auto N = in.rows(), M = in.cols();
    if (out.rows() != N || out.cols() != M)
      throw std::logic_error("ReLU::Forward: output shape mismatch");

    m_Rows = N, m_Cols = M;
    m_Mask.assign(N * M, std::uint8_t{0});
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

    const T *up = dH.data();
    T *down = dX.data();
    const std::size_t total = m_Rows * m_Cols;

    for (std::size_t idx = 0; idx < total; idx++)
      down[idx] = m_Mask[idx] ? up[idx] : T{0};
  }

  void ZeroGrads() override {}
  void GradientDescentStep(float) override {}

private:
  std::size_t m_Rows = 0, m_Cols = 0;
  std::vector<std::uint8_t> m_Mask;
};
} // namespace Logos::NeuralNet
