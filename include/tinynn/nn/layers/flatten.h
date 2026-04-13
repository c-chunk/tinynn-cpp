#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <tinynn/nn/layer.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

template <class T>
class Flatten final : public Layer<T> {
 public:
  Shape output_shape(const Shape& input_shape) const override {
    if (input_shape.rank() < 2) {
      throw std::invalid_argument(
          "tinynn::Flatten::output_shape: rank must be >= 2");
    }
    const SizeType B = input_shape.dim(0);
    SizeType feat = 1;
    for (SizeType i = 1; i < input_shape.rank(); ++i) {
      feat *= input_shape.dim(i);
    }
    return Shape{std::vector<SizeType>{B, feat}};
  }

  void forward(ConstTensorView<T> x, TensorView<T> y) override {
    const Shape out = output_shape(x.shape());
    if (y.shape() != out) {
      throw std::invalid_argument(
          "tinynn::Flatten::forward: y shape mismatch");
    }
    std::copy(x.data(), x.data() + x.size(), y.data());
  }

  void backward(ConstTensorView<T> dy, TensorView<T> dx) override {
    if (dy.size() != dx.size()) {
      throw std::invalid_argument(
          "tinynn::Flatten::backward: volume mismatch");
    }
    std::copy(dy.data(), dy.data() + dy.size(), dx.data());
  }
};

}  // namespace tinynn
