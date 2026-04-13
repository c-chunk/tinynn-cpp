#pragma once

#include <stdexcept>

#include <tinynn/nn/layer.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

// ReLU layer: y = max(0, x)
template <class T>
class ReLU final : public Layer<T> {
 public:
  ReLU() = default;

  [[nodiscard]] double last_pos_ratio() const noexcept {
    return last_pos_ratio_;
  }

  Shape output_shape(const Shape& input_shape) const override {
    return input_shape;  // same shape
  }

  void forward(ConstTensorView<T> x, TensorView<T> y) override {
    ensure_same_shape(x, y);
    last_x_ = x;

    const SizeType n = x.size();
    SizeType pos = 0;
    for (SizeType i = 0; i < n; ++i) {
      const T v = x[i];
      if (v > static_cast<T>(0)) ++pos;
      y[i] = (v > static_cast<T>(0)) ? v : static_cast<T>(0);
    }
    last_pos_ratio_ =
        (n == 0) ? 0.0
                 : (static_cast<double>(pos) / static_cast<double>(n));
  }

  void backward(ConstTensorView<T> dy, TensorView<T> dx) override {
    ensure_same_shape(dy, dx);
    ensure_cached_input();

    const SizeType n = dy.size();
    for (SizeType i = 0; i < n; ++i) {
      // ReLU'(x) = 1 if x>0 else 0
      dx[i] = (last_x_[i] > static_cast<T>(0)) ? dy[i] : static_cast<T>(0);
    }

    clear_cache();
  }

  void clear_cache() noexcept { last_x_ = ConstTensorView<T>(); }

 private:
  static void ensure_same_shape(ConstTensorView<T> a, TensorView<T> b) {
    if (a.size() != b.size()) {
      throw std::invalid_argument(
          "tinynn::ReLU: shape mismatch (volume differs)");
    }
  }

  void ensure_cached_input() const {
    if (last_x_.data() == nullptr) {
      throw std::invalid_argument(
          "tinynn::ReLU::backward: missing cached input (call forward first)");
    }
    if (last_x_.size() == 0) {
      // empty tensor is allowed; nothing to do
      return;
    }
  }

  ConstTensorView<T> last_x_{};

  double last_pos_ratio_ = 0.0;
};

}  // namespace tinynn
