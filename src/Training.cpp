#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "Math/Functions.hpp"
#include "Training.hpp"

namespace Logos::NeuralNet {
TrainModel::TrainModel(Memory::Arena &arena)
    : m_Arena(&arena), m_Model(arena, INPUT_LAYER, HIDDEN, OUTPUT_LAYER, m_RNG),
      m_LearningRate(LEARNING_RATE), m_datasetInfo(arena) {

  m_Order.resize(m_datasetInfo.GetTrainImgs().rows());
  std::iota(m_Order.begin(), m_Order.end(), 0);

  std::cout << "Train: N=" << m_datasetInfo.GetTrainImgs().rows()
            << " | Test: N=" << m_datasetInfo.GetTestImgs().rows() << '\n';
}

void TrainModel::run() {
  std::vector<uint8_t> yb;
  for (std::uint32_t ep = 1; ep <= EPOCHS; ep++) {

    Memory::ScratchArena scratchArena(*m_Arena);
    linalg::ScratchArenaMatrix<float> Xb(scratchArena, BATCH_SIZE,
                                         m_datasetInfo.GetTrainImgs().cols());
    linalg::ScratchArenaMatrix<float> logits(scratchArena, BATCH_SIZE,
                                             OUTPUT_LAYER);

    std::shuffle(m_Order.begin(), m_Order.end(), m_RNG);

    float loss_acc = 0.0;
    std::size_t steps = 0;

    const std::size_t sz = (m_Order.size() / BATCH_SIZE) * BATCH_SIZE;
    for (std::size_t start = 0; start < sz; start += BATCH_SIZE) {
      make_batch(m_datasetInfo.GetTrainImgs().cview(),
                 m_datasetInfo.GetTrainLabels(), m_Order, start, BATCH_SIZE,
                 Xb.view(), yb);

      const float loss = m_Model.TrainStep(Xb.cview(), yb, m_LearningRate);
      loss_acc += loss;
      steps++;

      if (steps % 500 == 0) {
        static std::uniform_int_distribution<std::size_t> dist_idx(
            0, yb.size() - 1);
        m_datasetInfo.show_prediction(scratchArena, logits.view(), m_Model,
                                      m_datasetInfo.GetTrainImgs().cview(),
                                      m_datasetInfo.GetTrainLabels(),
                                      m_Order[start + dist_idx(m_RNG)]);
      }
    }

    std::uint32_t correct = 0, total = 0;
    linalg::ScratchArenaMatrix<float> Xt(scratchArena, BATCH_SIZE,
                                         m_datasetInfo.GetTestImgs().cols());
    std::vector<uint8_t> yt;

    const std::size_t test_sz =
        (m_datasetInfo.GetTestImgs().rows() / BATCH_SIZE) * BATCH_SIZE;
    for (std::size_t start = 0; start < test_sz; start += BATCH_SIZE) {
      std::vector<std::size_t> test_idx(BATCH_SIZE);
      std::iota(test_idx.begin(), test_idx.end(), start);

      make_batch(m_datasetInfo.GetTestImgs().cview(),
                 m_datasetInfo.GetTestLabels(), test_idx, 0, BATCH_SIZE,
                 Xt.view(), yt);

      m_Model.Forward(Xt.cview(), logits.view());

      for (std::size_t i = 0; i < BATCH_SIZE; i++) {
        const std::size_t pred = ArgmaxRow<float>(logits.cview(), i);
        if (pred == yt[i])
          correct++;
        total++;
      }
    }

    const float test_acc =
        (total == 0) ? 0.0f : static_cast<float>(correct) / total;
    const float mean_loss = (steps == 0) ? 0.0f : loss_acc / steps;

    std::cout << "Epoch " << ep << " done | lr=" << m_LearningRate
              << " mean_loss=" << mean_loss << " test_acc=" << test_acc << '\n';

    m_LearningRate *= LEARNING_RATE_DECAY;
  }
}

void TrainModel::make_batch(linalg::MatrixView<const float> imgs,
                            const std::vector<std::uint8_t> &labels,
                            const std::vector<std::size_t> &indices,
                            std::size_t start, std::size_t batch_size,
                            linalg::MatrixView<float> Xb,
                            std::vector<std::uint8_t> &yb) {

  const auto D = imgs.cols(), N = indices.size(),
             end = std::min(start + batch_size, N), B = end - start;

  if (B == 0)
    throw std::logic_error("make_batch: empty batch");

  if (Xb.rows() < B || Xb.cols() != D)
    throw std::logic_error("make_batch: Xb has wrong shape");

  yb.resize(B);

  for (std::size_t i = 0; i < B; i++) {
    const auto idx = indices[start + i];
    if (idx >= imgs.rows() || idx >= labels.size())
      throw std::logic_error("make_batch: index out of range");

    yb[i] = labels[idx];

    for (std::size_t j = 0; j < D; j++)
      Xb(i, j) = imgs(idx, j);
  }
}
} // namespace Logos::NeuralNet
