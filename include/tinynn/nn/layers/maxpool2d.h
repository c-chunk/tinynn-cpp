#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <tinynn/nn/layer.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

struct MaxPool2dOptions {
  SizeType kernel_h = 2;
  SizeType kernel_w = 2;
  SizeType stride_h = 2;
  SizeType stride_w = 2;
};

template <class T>
class MaxPool2d : public Layer<T> {
 public:
  explicit MaxPool2d(MaxPool2dOptions opt)
      : kH_(opt.kernel_h),
        kW_(opt.kernel_w),
        sH_(opt.stride_h),
        sW_(opt.stride_w) {
    if (kH_ == 0 || kW_ == 0) {
      throw std::invalid_argument("MaxPool2d: kernel must be > 0");
    }
    if (sH_ == 0 || sW_ == 0) {
      throw std::invalid_argument("MaxPool2d: stride must be > 0");
    }
  }

  // Convenience constructor: kernel_size and stride (square)
  explicit MaxPool2d(SizeType kernel_size, SizeType stride)
      : MaxPool2d(MaxPool2dOptions{kernel_size, kernel_size, stride, stride}) {}

  Shape output_shape(const Shape& in) const override {
    if (in.rank() != 4) {
      throw std::invalid_argument("MaxPool2d: input must be NCHW");
    }

    const SizeType N = in.dim(0);
    const SizeType C = in.dim(1);
    const SizeType H = in.dim(2);
    const SizeType W = in.dim(3);

    if (H < kH_ || W < kW_) {
      throw std::invalid_argument("MaxPool2d: kernel larger than input");
    }

    const SizeType Ho = (H - kH_) / sH_ + 1;
    const SizeType Wo = (W - kW_) / sW_ + 1;

    return Shape{{N, C, Ho, Wo}};
  }

  void forward(ConstTensorView<T> x, TensorView<T> y) override {
    const auto& s = x.shape();

    if (s.rank() != 4) {
      throw std::invalid_argument("MaxPool2d::forward: input must be rank 4");
    }

    const SizeType N = s.dim(0);
    const SizeType C = s.dim(1);
    const SizeType H = s.dim(2);
    const SizeType W = s.dim(3);

    const SizeType Ho = (H - kH_) / sH_ + 1;
    const SizeType Wo = (W - kW_) / sW_ + 1;

    max_index_.resize(N * C * Ho * Wo);

    SizeType idx = 0;

    for (SizeType n = 0; n < N; ++n) {
      for (SizeType c = 0; c < C; ++c) {
        for (SizeType oh = 0; oh < Ho; ++oh) {
          for (SizeType ow = 0; ow < Wo; ++ow) {

            const SizeType ih = oh * sH_;
            const SizeType iw = ow * sW_;

            T max_val = x(n, c, ih, iw);
            SizeType max_k = 0;

            SizeType k = 0;

            for (SizeType kh = 0; kh < kH_; ++kh) {
              for (SizeType kw = 0; kw < kW_; ++kw) {

                const SizeType h = ih + kh;
                const SizeType w = iw + kw;

                const T v = x(n, c, h, w);

                if (v > max_val) {
                  max_val = v;
                  max_k = k;
                }

                ++k;
              }
            }

            y(n, c, oh, ow) = max_val;
            max_index_[idx++] = max_k;
          }
        }
      }
    }
  }

  void backward(ConstTensorView<T> dy, TensorView<T> dx) override {
    const auto& s = dx.shape();

    const SizeType N = s.dim(0);
    const SizeType C = s.dim(1);
    const SizeType H = s.dim(2);
    const SizeType W = s.dim(3);

    const SizeType Ho = (H - kH_) / sH_ + 1;
    const SizeType Wo = (W - kW_) / sW_ + 1;

    std::fill(dx.data(), dx.data() + dx.size(), static_cast<T>(0));

    SizeType idx = 0;

    for (SizeType n = 0; n < N; ++n) {
      for (SizeType c = 0; c < C; ++c) {
        for (SizeType oh = 0; oh < Ho; ++oh) {
          for (SizeType ow = 0; ow < Wo; ++ow) {

            const SizeType ih = oh * sH_;
            const SizeType iw = ow * sW_;

            const SizeType k = max_index_[idx++];

            const SizeType kh = k / kW_;
            const SizeType kw = k % kW_;

            dx(n, c, ih + kh, iw + kw) += dy(n, c, oh, ow);
          }
        }
      }
    }
  }

 private:
  SizeType kH_;
  SizeType kW_;
  SizeType sH_;
  SizeType sW_;

  std::vector<SizeType> max_index_;
};

}  // namespace tinynn
