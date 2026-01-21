# Logos-ML

A **C++ neural network framework built from scratch** focused on **explicit memory management**, **simplicity** and **clarity**, with an implementation of training a multilayer perceptron on the **MNIST** dataset.

---

## Overview

Logos-ML explores the internal mechanics of machine learning systems by implementing core components manually:

- memory buffers  
- matrix abstractions  
- kernels  
- neural network layers  
- loss functions  
- training

The project avoids external ML libraries.

---

## Features

- Custom **aligned memory allocation**
- Move-only **matrix abstraction**
- **Linear Layers**
- **ReLU** activation
- **Softmax + Cross-Entropy** loss
- Mini-batch **gradient descent**
- **MNIST classification** example

---

## Abstractions

The implementation is performance-conscious and minimal.

### Buffer
- Owns aligned raw memory
- Provides explicit lifetime control

### Matrix
- Built directly on top of `Buffer`

### Layers
- Stores only necessary state for backpropagation
- Gradients are computed by hand

---

## Architecture Overview

High-level components:

- **Buffer** — aligned memory management  
- **Matrix** — rank-2 Tensor abstraction  
- **Linear** — layer with weights and biases  
- **ReLU** — activation layer  
- **Softmax / CrossEntropy** — output normalization and loss  
- **MLP_Hardcoded** — multilayer perceptron model  
- **TrainModel** — data loading, batching, training loop  


---

## Example: MNIST Model

The included model trains a multilayer perceptron with:

- Input layer: **784** (28×28)
- Hidden layer: **256** neurons (ReLU)
- Output layer: **10** logits (classes 0–9)
- Optimizer: **Gradient Descent** (mini-batch)
- Loss: **Softmax + Cross-Entropy**

Training, inference, and evaluation are implemented explicitly without high-level framework abstractions.

---

## Build

This project uses **CMake**.

From the project root:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

---

## MNIST Setup

To run MNIST training, use the provided Python helper script to download and prepare the dataset.

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
./Logos
```
(Windows: Logos.exe)

---

## Roadmap
Planned changes:
- SIMD-optimized kernels
- Additional activation functions
- Unit testing for numerical kernels and layers
- Improved numerical stability
- Backend abstraction for hardware acceleration
- Convolution layers
