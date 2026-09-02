#include "Core/Tensor.hpp"
#include "Data/DataLoader.hpp"
#include "Modules/Linear.hpp"
#include "Modules/Parameter.hpp"
#include "Ops/Loss.hpp"
#include "Ops/Matmul.hpp"
#include "Ops/ReLU.hpp"
#include "Ops/Softmax.hpp"
#include "Optimizer/SGD.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string_view>

namespace {
int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void near(float actual, float expected, float tolerance,
          std::string_view message) {
  check(std::abs(actual - expected) <= tolerance, message);
}

float cross_entropy_value(ml::core::Tensor &logits,
                          const ml::core::Tensor &labels) {
  ml::core::Tensor probabilities(logits.shape());
  ml::core::Tensor loss(ml::core::Shape{});
  ml::ops::cross_entropy(logits, labels, probabilities, loss);
  return loss.data<float>()[0];
}

float linear_objective(ml::core::Linear &layer, const ml::core::Tensor &input,
                       const ml::core::Tensor &upstream) {
  ml::core::Tensor output(layer.output_shape(input.shape()), input.dtype());
  layer.forward(input, output);

  float value = 0.0f;
  for (int i = 0; i < output.num_elements(); ++i)
    value += output.data<float>()[i] * upstream.data<float>()[i];
  return value;
}

void test_tensor_copy_and_clone() {
  ml::core::Tensor tensor({2});
  tensor.data<float>()[0] = 1.0f;
  tensor.data<float>()[1] = 2.0f;

  auto alias = tensor;
  alias.data<float>()[0] = 3.0f;
  near(tensor.data<float>()[0], 3.0f, 0.0f,
       "Tensor copies should share storage");

  auto clone = tensor.clone();
  clone.data<float>()[0] = 4.0f;
  near(tensor.data<float>()[0], 3.0f, 0.0f,
       "Tensor::clone should make an independent copy");
  near(tensor.operator()<float>(1), 2.0f, 0.0f,
       "typed tensor indexing should return the requested element");
}

void test_matmul() {
  ml::core::Tensor a({2, 3}), b({3, 2}), out({2, 2});
  const float av[] = {1, 2, 3, 4, 5, 6};
  const float bv[] = {7, 8, 9, 10, 11, 12};
  std::copy_n(av, 6, a.data<float>());
  std::copy_n(bv, 6, b.data<float>());

  ml::ops::matmul(ml::ops::Transpose::No, ml::ops::Transpose::No, true, a, b,
                  out);
  const float expected[] = {58, 64, 139, 154};
  for (int i = 0; i < 4; ++i)
    near(out.data<float>()[i], expected[i], 1e-5f, "matrix multiplication");
}

void test_softmax_and_cross_entropy() {
  ml::core::Tensor logits({2, 3});
  const float values[] = {1000, 0, -1000, -1000, 0, 1000};
  std::copy_n(values, 6, logits.data<float>());
  ml::core::Tensor probabilities({2, 3});
  ml::ops::softmax(logits, probabilities);
  for (int row = 0; row < 2; ++row) {
    float sum = 0;
    for (int col = 0; col < 3; ++col)
      sum += probabilities.data<float>()[row * 3 + col];
    near(sum, 1.0f, 1e-6f, "each softmax row should sum to one");
  }

  ml::core::Tensor labels({2}, ml::core::DType::Int32);
  labels.data<int>()[0] = 0;
  labels.data<int>()[1] = 2;
  const float loss = cross_entropy_value(logits, labels);
  check(std::isfinite(loss),
        "cross entropy should remain finite for large logits");
  near(loss, 0.0f, 1e-6f,
       "confident correct predictions should have zero loss");
}

void test_cross_entropy_gradient() {
  ml::core::Tensor logits({2, 3});
  const float values[] = {0.2f, -0.4f, 1.1f, -0.3f, 0.7f, 0.1f};
  std::copy_n(values, 6, logits.data<float>());
  ml::core::Tensor labels({2}, ml::core::DType::Int32);
  labels.data<int>()[0] = 2;
  labels.data<int>()[1] = 1;

  ml::core::Tensor probabilities({2, 3}), loss(ml::core::Shape{}), grad({2, 3});
  ml::ops::cross_entropy(logits, labels, probabilities, loss);
  ml::ops::cross_entropy_backward(probabilities, labels, grad);

  constexpr float epsilon = 1e-3f;
  for (int i = 0; i < logits.num_elements(); ++i) {
    const float original = logits.data<float>()[i];
    logits.data<float>()[i] = original + epsilon;
    const float plus = cross_entropy_value(logits, labels);
    logits.data<float>()[i] = original - epsilon;
    const float minus = cross_entropy_value(logits, labels);
    logits.data<float>()[i] = original;
    near(grad.data<float>()[i], (plus - minus) / (2 * epsilon), 1e-4f,
         "cross-entropy analytical gradient should match finite differences");
  }
}

void test_relu_backward() {
  ml::core::Tensor input({4}), output({4}), upstream({4}), gradient({4});
  const float values[] = {-2.0f, -0.1f, 0.2f, 3.0f};
  const float upstream_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
  std::copy_n(values, 4, input.data<float>());
  std::copy_n(upstream_values, 4, upstream.data<float>());
  ml::ops::relu(input, output);
  ml::ops::relu_backward(upstream, input, gradient);
  const float expected[] = {0, 0, 3, 4};
  for (int i = 0; i < 4; ++i)
    near(gradient.data<float>()[i], expected[i], 0.0f, "ReLU backward mask");
}

void test_linear_backward() {
  ml::core::Linear layer(2, 3);
  auto parameters = layer.own_parameters();
  auto *weight = parameters[0];
  auto *bias = parameters[1];
  const float weight_values[] = {0.2f, -0.3f, 0.5f, 0.7f, 0.1f, -0.4f};
  const float bias_values[] = {0.05f, -0.2f, 0.3f};
  std::copy_n(weight_values, 6, weight->data.data<float>());
  std::copy_n(bias_values, 3, bias->data.data<float>());

  ml::core::Tensor input({2, 2});
  const float input_values[] = {0.6f, -1.2f, 0.4f, 0.8f};
  std::copy_n(input_values, 4, input.data<float>());
  ml::core::Tensor upstream({2, 3});
  const float upstream_values[] = {0.3f, -0.7f, 0.2f, -0.5f, 0.4f, 0.9f};
  std::copy_n(upstream_values, 6, upstream.data<float>());

  // backward() computes gradients of dot(forward(input), upstream).
  ml::core::Tensor output(layer.output_shape(input.shape()));
  layer.forward(input, output);
  ml::core::Tensor input_gradient(input.shape());
  layer.backward(upstream, input_gradient);

  constexpr float epsilon = 1e-3f;
  for (int i = 0; i < input.num_elements(); ++i) {
    const float original = input.data<float>()[i];
    input.data<float>()[i] = original + epsilon;
    const float plus = linear_objective(layer, input, upstream);
    input.data<float>()[i] = original - epsilon;
    const float minus = linear_objective(layer, input, upstream);
    input.data<float>()[i] = original;
    near(input_gradient.data<float>()[i], (plus - minus) / (2 * epsilon), 1e-4f,
         "Linear input gradient should match finite differences");
  }

  for (int i = 0; i < weight->data.num_elements(); ++i) {
    const float original = weight->data.data<float>()[i];
    weight->data.data<float>()[i] = original + epsilon;
    const float plus = linear_objective(layer, input, upstream);
    weight->data.data<float>()[i] = original - epsilon;
    const float minus = linear_objective(layer, input, upstream);
    weight->data.data<float>()[i] = original;
    near(weight->grad->data<float>()[i], (plus - minus) / (2 * epsilon), 1e-4f,
         "Linear weight gradient should match finite differences");
  }

  for (int i = 0; i < bias->data.num_elements(); ++i) {
    const float original = bias->data.data<float>()[i];
    bias->data.data<float>()[i] = original + epsilon;
    const float plus = linear_objective(layer, input, upstream);
    bias->data.data<float>()[i] = original - epsilon;
    const float minus = linear_objective(layer, input, upstream);
    bias->data.data<float>()[i] = original;
    near(bias->grad->data<float>()[i], (plus - minus) / (2 * epsilon), 1e-4f,
         "Linear bias gradient should match finite differences");
  }
}

void test_sgd() {
  ml::core::Parameter parameter(ml::core::Tensor({2}));
  parameter.data.data<float>()[0] = 1.0f;
  parameter.data.data<float>()[1] = -2.0f;
  parameter.grad = ml::core::Tensor({2});
  parameter.grad->data<float>()[0] = 0.5f;
  parameter.grad->data<float>()[1] = -0.25f;
  ml::optim::SGD<float> optimizer({&parameter}, 0.1f, 0.0f, 0.2f);
  optimizer.step();
  near(parameter.data.data<float>()[0], 0.93f, 1e-6f,
       "SGD should apply gradient and L2 penalty");
  near(parameter.data.data<float>()[1], -1.935f, 1e-6f,
       "SGD should apply signed gradient and L2 penalty");
  optimizer.zero_grad();
  near(parameter.grad->data<float>()[0], 0.0f, 0.0f,
       "SGD::zero_grad should clear gradients");
}

void test_data_loader() {
  ml::core::Tensor images({4, 2});
  ml::core::Tensor labels({4}, ml::core::DType::Int32);
  for (int i = 0; i < 8; ++i)
    images.data<float>()[i] = static_cast<float>(i);
  for (int i = 0; i < 4; ++i)
    labels.data<int>()[i] = i;

  ml::data::DataLoader loader(images, labels, 2, false);
  ml::data::Batch batch;
  check(loader.next(batch), "DataLoader should produce its first batch");
  near(batch.images.data<float>()[2], 2.0f, 0.0f,
       "DataLoader should preserve row order when shuffle is disabled");
  check(loader.next(batch), "DataLoader should produce its second batch");
  check(!loader.next(batch), "DataLoader should stop after a full epoch");
  loader.reset();
  check(loader.next(batch), "DataLoader::reset should start a new epoch");
  check(batch.labels.data<int>()[0] == 0,
        "unshuffled DataLoader reset should preserve order");
}
} // namespace

int main() {
  test_tensor_copy_and_clone();
  test_matmul();
  test_softmax_and_cross_entropy();
  test_cross_entropy_gradient();
  test_relu_backward();
  test_linear_backward();
  test_sgd();
  test_data_loader();

  if (failures != 0) {
    std::cerr << failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "All tests passed\n";
}
