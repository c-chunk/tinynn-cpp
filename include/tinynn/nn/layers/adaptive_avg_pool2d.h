#pragma once

#include <stdexcept>

#include <tinynn/nn/layer.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

// AdaptiveAvgPool2d
// Input : [N, C, H_in, W_in]
// Output: [N, C, H_out, W_out]
//
// For each output cell (oh, ow), averages the corresponding input region:
//
//   h_start = floor( oh      * H_in / H_out )
//   h_end   = ceil ( (oh + 1)* H_in / H_out )
//   w_start = floor( ow      * W_in / W_out )
//   w_end   = ceil ( (ow + 1)* W_in / W_out )
//
// and computes the mean over that region.
template <class T>
class AdaptiveAvgPool2d final : public Layer<T> {
 public:
  AdaptiveAvgPool2d(SizeType out_h, SizeType out_w)
      : out_h_(out_h), out_w_(out_w) {
    if (out_h_ == 0 || out_w_ == 0) {
      throw std::invalid_argument(
          "tinynn::AdaptiveAvgPool2d: output size must be > 0");
    }
  }

  [[nodiscard]] SizeType out_h() const noexcept { return out_h_; }
  [[nodiscard]] SizeType out_w() const noexcept { return out_w_; }

  Shape output_shape(const Shape& in) const override {
    if (in.rank() != 4) {
      throw std::invalid_argument(
          "tinynn::AdaptiveAvgPool2d::output_shape: "
          "input must be rank-4 (NCHW)");
    }
    return Shape{{in.dim(0), in.dim(1), out_h_, out_w_}};
  }

  void forward(ConstTensorView<T> x, TensorView<T> y) override {
    if (x.shape().rank() != 4) {
      throw std::invalid_argument(
          "tinynn::AdaptiveAvgPool2d::forward: "
          "input must be rank-4 (NCHW)");
    }

    const Shape out = output_shape(x.shape());
    if (y.shape() != out) {
      throw std::invalid_argument(
          "tinynn::AdaptiveAvgPool2d::forward: y shape mismatch");
    }

    const SizeType N = x.shape().dim(0);
    const SizeType C = x.shape().dim(1);
    const SizeType H_in = x.shape().dim(2);
    const SizeType W_in = x.shape().dim(3);

    saved_input_shape_ = x.shape();

    for (SizeType n = 0; n < N; ++n) {
      for (SizeType c = 0; c < C; ++c) {
        for (SizeType oh = 0; oh < out_h_; ++oh) {
          const SizeType h_start = start_index_(oh, out_h_, H_in);
          const SizeType h_end   = end_index_(oh, out_h_, H_in);

          for (SizeType ow = 0; ow < out_w_; ++ow) {
            const SizeType w_start = start_index_(ow, out_w_, W_in);
            const SizeType w_end   = end_index_(ow, out_w_, W_in);

            const SizeType area = (h_end - h_start) * (w_end - w_start);
            if (area == 0) {
              throw std::logic_error(
                  "tinynn::AdaptiveAvgPool2d::forward: empty pooling region");
            }

            T sum = static_cast<T>(0);
            for (SizeType h = h_start; h < h_end; ++h) {
              for (SizeType w = w_start; w < w_end; ++w) {
                sum += x(n, c, h, w);
              }
            }

            y(n, c, oh, ow) = sum / static_cast<T>(area);
          }
        }
      }
    }
  }

  void backward(ConstTensorView<T> dy, TensorView<T> dx) override {
    if (saved_input_shape_.rank() == 0) {
      throw std::logic_error(
          "tinynn::AdaptiveAvgPool2d::backward: "
          "called before forward");
    }

    if (dx.shape() != saved_input_shape_) {
      throw std::invalid_argument(
          "tinynn::AdaptiveAvgPool2d::backward: dx shape mismatch");
    }

    const Shape expected_dy = output_shape(saved_input_shape_);
    if (dy.shape() != expected_dy) {
      throw std::invalid_argument(
          "tinynn::AdaptiveAvgPool2d::backward: dy shape mismatch");
    }

    const SizeType N = dx.shape().dim(0);
    const SizeType C = dx.shape().dim(1);
    const SizeType H_in = dx.shape().dim(2);
    const SizeType W_in = dx.shape().dim(3);

    // Initialize dx to zero.
    for (SizeType i = 0; i < dx.size(); ++i) {
      dx[i] = static_cast<T>(0);
    }

    for (SizeType n = 0; n < N; ++n) {
      for (SizeType c = 0; c < C; ++c) {
        for (SizeType oh = 0; oh < out_h_; ++oh) {
          const SizeType h_start = start_index_(oh, out_h_, H_in);
          const SizeType h_end   = end_index_(oh, out_h_, H_in);

          for (SizeType ow = 0; ow < out_w_; ++ow) {
            const SizeType w_start = start_index_(ow, out_w_, W_in);
            const SizeType w_end   = end_index_(ow, out_w_, W_in);

            const SizeType area = (h_end - h_start) * (w_end - w_start);
            if (area == 0) {
              throw std::logic_error(
                  "tinynn::AdaptiveAvgPool2d::backward: empty pooling region");
            }

            const T g = dy(n, c, oh, ow) / static_cast<T>(area);

            for (SizeType h = h_start; h < h_end; ++h) {
              for (SizeType w = w_start; w < w_end; ++w) {
                dx(n, c, h, w) += g;
              }
            }
          }
        }
      }
    }

    clear_cache_();
  }

 private:
  static SizeType start_index_(SizeType out_idx,
                               SizeType out_size,
                               SizeType in_size) {
    // floor(out_idx * in_size / out_size)
    return (out_idx * in_size) / out_size;
  }

  static SizeType end_index_(SizeType out_idx,
                             SizeType out_size,
                             SizeType in_size) {
    // ceil((out_idx + 1) * in_size / out_size)
    return ((out_idx + 1) * in_size + out_size - 1) / out_size;
  }

  void clear_cache_() noexcept {
    saved_input_shape_ = Shape{};
  }

 private:
  SizeType out_h_;
  SizeType out_w_;

  Shape saved_input_shape_{};
};

}  // namespace tinynn
