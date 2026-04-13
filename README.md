# tinynn-cpp

A tiny neural network framework written in **C++20**\
for learning, experimentation, and understanding how deep learning
libraries work.

tinynn provides a compact implementation of tensors, layers, optimizers,
and a training loop, with a focus on **clarity and educational value**
rather than production performance.

------------------------------------------------------------------------

## ✨ Features

-   Header-only design (no build step required for the core)
-   Contiguous Tensor implementation (row-major, NCHW for CNN)
-   Simple Sequential model API
-   Core layers (Affine, ReLU, Sigmoid)
-   CNN layers (Conv2d, MaxPool2d, GlobalAveragePool2d, etc.)
-   Batch normalization and dropout (including Dropout2d)
-   Optimizers (SGD, Momentum, Adam, AdamW, RMSProp)
-   Trainer with callback system
-   Checkpoint save/load (full state roundtrip)
-   QMNIST dataset support

------------------------------------------------------------------------

## 🎯 Why tinynn?

Modern frameworks like **PyTorch** or **TensorFlow** are powerful but
internally complex.

tinynn is designed as a **small, readable C++ framework** that allows
you to:

-   understand how neural networks are implemented
-   study training pipelines and backpropagation
-   experiment with architectures and optimizers
-   build your own ML components from scratch

------------------------------------------------------------------------

## 🧠 Design Goals

tinynn follows a few simple principles:

-   **Small and readable codebase**
-   **Header-only architecture**
-   **Minimal external dependencies**
-   **Modern C++ (C++20)**
-   **Clarity over performance**

------------------------------------------------------------------------

## 🚧 Project Status

tinynn is currently an **experimental project**.

------------------------------------------------------------------------

## 📁 Directory Structure

    include/tinynn   core library (header-only)
    examples         example programs
    tests            unit and smoke tests
    data/qmnist      dataset location (not included)

------------------------------------------------------------------------

## 🔧 Requirements

-   C++20 compatible compiler
-   CMake 3.16+

------------------------------------------------------------------------

## 🛠️ Build

``` bash
git clone https://github.com/c-chunk/tinynn-cpp
cd tinynn-cpp

mkdir build
cd build
cmake ..
cmake --build .
```

------------------------------------------------------------------------

## ▶️ Run Examples

``` bash
./example_minimal
./example_qmnist_cnn
```

------------------------------------------------------------------------

## 📦 Dataset (QMNIST)

QMNIST data is **not included** in the repository.

Place files under:

    data/qmnist/

------------------------------------------------------------------------

## 🧪 Example

``` cpp
#include <tinynn/tinynn.h>

using namespace tinynn;

int main()
{
  Sequential<float> model;

  model.add(Affine<float>(784, 128));
  model.add(ReLU<float>());
  model.add(Affine<float>(128, 10));

  return 0;
}
```

------------------------------------------------------------------------

## 📄 License

MIT License\
Copyright (c) 2026 c-chunk
