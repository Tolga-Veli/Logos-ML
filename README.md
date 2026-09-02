# Logos-ML
A small **C++23 neural-network framework built from scratch**, implementing an MNIST multilayer-perceptron example.


## Overview
Logos-ML implements core machine learning concepts from scratch:

- reference-counted tensor storage
- shapes, strides, and typed tensor handles
- CPU kernels backed by BLAS where appropriate
- neural-network modules and manually implemented backpropagation (doesn't support residual connections)
- loss functions, optimization, batching, and training

The project does not depend on an external machine-learning framework. It does
use BLAS for fast matrix and vector operations.


## Features
- Cheaply copyable, intrusive-reference-counted **Tensor** handles
- Explicit **Shape**, **Strides**, storage, dtype, and device abstractions
- BLAS-backed matrix multiplication and vector operations
- **Linear** layers and **Sequential** composition
- **ReLU** activation
- Numerically stable **Softmax + Cross-Entropy** loss
- Mini-batch **SGD**, with optional momentum and coupled L2 regularization
- CTest-based numerical and gradient tests
- **MNIST classification** example


## Abstractions
The implementation is performance-conscious and minimal.

### Storage and Tensor

- `Storage` owns a device allocation.
- `TensorImpl` owns shape, stride, offset, dtype, and shared storage metadata.
- `Tensor` is a cheaply copyable intrusive-reference-counted handle. Ordinary
  copies share storage; `clone()` performs a deep copy.
- `MatrixView`, `VectorView`, and `ScalarView` adapt tensors to kernel APIs.

### Modules

- `Module` provides forward/backward and parameter traversal interfaces.
- `Sequential` composes modules.
- `Linear` and `ReLU` retain the state required for their backward passes.
- Parameter gradients are calculated explicitly rather than by an autograd
  engine.


## Architecture Overview
High-level components:

- `src/Core` — tensors, shapes, strides, dtypes, assertions, and logging
- `src/Memory` — device allocation, storage, and intrusive references
- `src/Ops` — public operations, views, dispatch, and CPU kernels
- `src/Modules` — parameters, layers, and sequential composition
- `src/Optimizer` — SGD
- `src/Data` — loading file data and mini-batch iteration
- `src/main.cpp` — MNIST model, training loop, evaluation, and rendering
- `tests` — backward-gradient, optimizer, tensor, and loader tests



## Example: MNIST Model
The included model trains a multilayer perceptron with:

- Input layer: **784** (28×28)
- Hidden layer: **256** neurons (ReLU)
- Output layer: **10** logits (classes 0–9)
- Optimizer: **Gradient Descent** (mini-batch)
- Loss: **Softmax + Cross-Entropy**

Training, inference, and evaluation are implemented explicitly without high-level framework abstractions.


## Build
This project uses **CMake**.

From the project root, using GCC or Clang:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

With a multi-configuration generator such as Visual Studio:

```bash
cmake -S . -B build
cmake --build build --config Release
```

BLAS and a C++23-capable compiler are required.

## Tests

Tests are enabled by default through CTest:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## MNIST Setup
To run MNIST training, use the provided Python helper script to download and prepare the dataset.
The Python script must be run in the same folder as the program executable.

### Requirements
- **Python** 3.9–3.12  
- **Python packages**
  - `tensorflow_datasets`
  - `numpy`

Install the required packages:
```bash
pip install tensorflow_datasets numpy
```

Run the helper script (from the directory containing the executable, typically build/):
```bash
python help2.py
```

Then run the program:
```bash
./ml-project
```


## Roadmap
Planned changes:
- Additional activation functions
- More comprehensive tests, including randomized property tests
- Adam and AdamW optimizers
- Model serialization
- Convolution layers
- CUDA
