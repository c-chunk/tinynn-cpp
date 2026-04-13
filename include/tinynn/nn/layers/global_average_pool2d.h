#pragma once

#include <stdexcept>

#include <tinynn/nn/layer.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

// GlobalAveragePool2d
// Input : [N, C, H, W]
// Output: [N, C, 1, 1]
//
// For each (n, c), averages over the entire HxW plane.
template <class T>
class GlobalAveragePool2d final : public Layer<T> {
 public:
  GlobalAveragePool2d() = default;

  Shape output_shape(const Shape& in) const override {
    if (in.rank() != 4) {
      throw std::invalid_argument(
          "tinynn::GlobalAveragePool2d::output_shape: "
          "input must be rank-4 (NCHW)");
    }
    return Shape{{in.dim(0), in.dim(1), 1, 1}};
  }

  void forward(ConstTensorView<T> x, TensorView<T> y) override {
    if (x.shape().rank() != 4) {
      throw std::invalid_argument(
          "tinynn::GlobalAveragePool2d::forward: "
          "input must be rank-4 (NCHW)");
    }

    const Shape out = output_shape(x.shape());
    if (y.shape() != out) {
      throw std::invalid_argument(
          "tinynn::GlobalAveragePool2d::forward: y shape mismatch");
    }

    const SizeType N = x.shape().dim(0);
    const SizeType C = x.shape().dim(1);
    const SizeType H = x.shape().dim(2);
    const SizeType W = x.shape().dim(3);

    const SizeType HW = H * W;
    if (HW == 0) {
      throw std::invalid_argument(
          "tinynn::GlobalAveragePool2d::forward: H*W must be > 0");
    }

    const T inv_hw = static_cast<T>(1) / static_cast<T>(HW);

    for (SizeType n = 0; n < N; ++n) {
      for (SizeType c = 0; c < C; ++c) {
        T sum = static_cast<T>(0);
        for (SizeType h = 0; h < H; ++h) {
          for (SizeType w = 0; w < W; ++w) {
            sum += x(n, c, h, w);
          }
        }
        y(n, c, 0, 0) = sum * inv_hw;
      }
    }
  }

  void backward(ConstTensorView<T> dy, TensorView<T> dx) override {
    if (dx.shape().rank() != 4) {
      throw std::invalid_argument(
          "tinynn::GlobalAveragePool2d::backward: "
          "dx must be rank-4 (NCHW)");
    }

    const Shape expected_dy = output_shape(dx.shape());
    if (dy.shape() != expected_dy) {
      throw std::invalid_argument(
          "tinynn::GlobalAveragePool2d::backward: dy shape mismatch");
    }

    const SizeType N = dx.shape().dim(0);
    const SizeType C = dx.shape().dim(1);
    const SizeType H = dx.shape().dim(2);
    const SizeType W = dx.shape().dim(3);

    const SizeType HW = H * W;
    if (HW == 0) {
      throw std::invalid_argument(
          "tinynn::GlobalAveragePool2d::backward: H*W must be > 0");
    }

    const T inv_hw = static_cast<T>(1) / static_cast<T>(HW);

    for (SizeType n = 0; n < N; ++n) {
      for (SizeType c = 0; c < C; ++c) {
        const T g = dy(n, c, 0, 0) * inv_hw;
        for (SizeType h = 0; h < H; ++h) {
          for (SizeType w = 0; w < W; ++w) {
            dx(n, c, h, w) = g;
          }
        }
      }
    }
  }
};

}  // namespace tinynn
