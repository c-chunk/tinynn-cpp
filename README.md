# tinynn-cpp

A tiny neural network framework in C++20.

tinynn-cpp is a lightweight, header-oriented deep learning framework designed for learning, experimentation, and small-scale projects.
It provides a clean and minimal API while supporting modern neural network components such as CNN layers, optimizers, and training utilities.

---

## ✨ Features

* **C++20-based design**
* **Header-oriented architecture** (minimal build friction)
* **Contiguous Tensor (row-major, NCHW for CNN)**
* **Sequential API**
* **Core Layers**

  * Affine (Fully Connected)
  * ReLU / Sigmoid
  * Conv2d / MaxPool2d
  * BatchNorm1d / BatchNorm2d
  * Dropout / Dropout2d
  * Flatten / GlobalAveragePool2d / AdaptiveAvgPool2d
* **Loss**

  * SoftmaxWithLoss (numerically stable)
* **Optimizers**

  * SGD / Momentum / RMSProp / Adam / AdamW
* **Trainer**

  * Training / Evaluation loop
  * Callback system
* **Checkpoint**

  * Model / Optimizer / Training state save & load
* **Dataset Support**

  * QMNIST example included

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/c-chunk/tinynn-cpp.git
cd tinynn-cpp
```

### 2. Build (CMake)

```bash
cmake -B build
cmake --build build
```

### 3. Run Example

```bash
./build/examples/qmnist_cnn
```

---

## 🧠 Example (MLP)

```cpp
#include <tinynn/nn/sequential.h>
#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/layers/relu.h>

using namespace tinynn;

Sequential<float> model;
model.add<Affine<float>>(784, 128);
model.add<ReLU<float>>();
model.add<Affine<float>>(128, 10);
```

---

## 🧠 Example (CNN)

```cpp
Sequential<float> model;

model.add<Conv2d<float>>(1, 32, 3, 3);
model.add<BatchNorm2d<float>>(32);
model.add<ReLU<float>>();
model.add<MaxPool2d<float>>(2, 2);

model.add<Flatten<float>>();
model.add<Affine<float>>(32 * 13 * 13, 10);
```

---

## 📊 Training Example

```cpp
Trainer<float> trainer(model, optimizer, loss);

trainer.fit(train_loader, test_loader, {
  .epochs = 5
});
```

---

## 📈 Sample Result

QMNIST CNN example:

* Accuracy: ~98% (after a few epochs)

---

## 📁 Project Structure

```
tinynn-cpp/
├── include/tinynn/        # Core library (header-oriented)
├── examples/              # Example programs (MLP / CNN)
├── tests/                 # Unit / integration tests
├── data/qmnist/           # Dataset (not included in repo)
├── CMakeLists.txt
```

---

## 🧩 Design Philosophy

tinynn-cpp aims to be:

* **Minimal but practical**
* **Readable and educational**
* **Easy to extend**
* **Close to the metal (no heavy abstraction)**

It is especially suitable for:

* Learning deep learning internals
* Experimenting with architectures
* Understanding training pipelines in C++

---

## ⚠️ Notes

* Tensors are **contiguous (row-major)**
* CNN uses **NCHW layout**
* TensorView supports **reshape-only view**
* No GPU support (CPU only)

---

## 🛠 Requirements

* C++20 compatible compiler (e.g. MSVC / Clang / GCC)
* CMake 3.16+

---

## 📄 License

MIT License

Copyright (c) c-chunk

---

## 🙌 Acknowledgements

This project is inspired by minimalist deep learning frameworks and aims to provide a clean C++ learning experience.
