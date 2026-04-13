#pragma once

#include <stdexcept>

#include <tinynn/nn/layer.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

// Sigmoid layer: y = 1 / (1 + exp(-x))
template <class T>
class Sigmoid final : public Layer<T> {
 public:
  Sigmoid() = default;

  Shape output_shape(const Shape& input_shape) const override {
    return input_shape;  // same shape
  }

  void forward(ConstTensorView<T> x, TensorView<T> y) override {
    ensure_same_shape(x, y);

    const SizeType n = x.size();
    for (SizeType i = 0; i < n; ++i) {
      const T v = x[i];
      y[i] = sigmoid(v);
    }

    // Cache output for backward (needs to remain valid until backward()).
    last_y_ = y;  // TensorView<T> -> TensorView<const T> implicit conversion
  }

  void backward(ConstTensorView<T> dy, TensorView<T> dx) override {
    ensure_same_shape(dy, dx);
    ensure_cached_output();

    const SizeType n = dy.size();
    for (SizeType i = 0; i < n; ++i) {
      const T yv = last_y_[i];
      dx[i] = dy[i] * yv * (static_cast<T>(1) - yv);
    }
  }

  void clear_cache() noexcept { last_y_ = ConstTensorView<T>(); }

 private:
  static void ensure_same_shape(ConstTensorView<T> a, TensorView<T> b) {
    if (a.size() != b.size()) {
      throw std::invalid_argument(
          "tinynn::Sigmoid: shape mismatch (volume differs)");
    }
  }

  void ensure_cached_output() const {
    if (last_y_.data() == nullptr) {
      throw std::invalid_argument(
          "tinynn::Sigmoid::backward: missing cached output (call forward first)");
    }
  }

  static T sigmoid(T x) {
    // Simple sigmoid implementation.
    // A branch-based formulation can improve numerical stability:
    //   if (x >= 0) 1/(1+exp(-x)) else exp(x)/(1+exp(x))
    using std::exp;
    return static_cast<T>(1) / (static_cast<T>(1) + static_cast<T>(exp(-x)));
  }

  ConstTensorView<T> last_y_{};
};

}  // namespace tinynn
