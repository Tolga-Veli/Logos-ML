#pragma once

#include <vector>

#include "Math/ArenaMatrix.hpp"
#include "Math/MatrixView.hpp"

#include "MLP.hpp"

namespace Logos::NeuralNet {
using NeuralNetwork = MLP_Hardcoded;

class MNIST {
public:
  MNIST(Memory::Arena &m_Arena);

  void draw_mnist_digit(const std::vector<float> &data);

  void show_prediction(Memory::ScratchArena &scratchArena,
                       linalg::MatrixView<float> logits, NeuralNetwork &model,
                       linalg::MatrixView<const float> imgs,
                       const std::vector<std::uint8_t> &labels,
                       std::size_t idx);

  linalg::MatrixView<float> GetTrainImgs() { return m_TrainImgs.view(); }
  linalg::MatrixView<const float> GetTrainImgsConst() {
    return m_TrainImgs.cview();
  }

  linalg::MatrixView<float> GetTestImgs() { return m_TestImgs.view(); }
  linalg::MatrixView<const float> GetTestImgsConst() {
    return m_TestImgs.cview();
  }

  const std::vector<std::uint8_t> &GetTrainLabels() const {
    return m_TrainLabels;
  }
  const std::vector<std::uint8_t> &GetTestLabels() const {
    return m_TestLabels;
  }

private:
  Memory::Arena *m_Arena;
  linalg::ArenaMatrix<float> m_TrainImgs, m_TestImgs;
  std::vector<std::uint8_t> m_TrainLabels, m_TestLabels;

  std::vector<float> get_mnist_image(linalg::MatrixView<const float> imgs,
                                     std::size_t idx);

  void load_images_mat(std::string path, std::size_t num, std::size_t rows,
                       std::size_t cols, linalg::MatrixView<float> out);

  std::vector<std::uint8_t> load_labels_mat(std::string path, std::size_t num);
};
} // namespace Logos::NeuralNet
