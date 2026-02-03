#include "Training.hpp"

int main() {
  Logos::Memory::Arena arena(1 * Logos::Memory::GiB);
  Logos::NeuralNet::TrainModel model{arena};
  model.run();
}
