#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include <tinynn/nn/init_policy.h>
#include <tinynn/nn/layer.h>
#include <tinynn/nn/parameter_view.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

struct Conv2dOptions {
  SizeType in_channels;
  SizeType out_channels;
  SizeType kernel_h;
  SizeType kernel_w;
  bool bias = true;

  SizeType stride_h = 1, stride_w = 1;
  SizeType pad_h = 0, pad_w = 0;
  SizeType dilation_h = 1, dilation_w = 1;
  SizeType groups = 1;
};

namespace detail {

static inline SizeType idx4(SizeType a, SizeType b, SizeType c, SizeType d,
                            SizeType B, SizeType C, SizeType D) noexcept {
  return ((a * B + b) * C + c) * D + d;
}

template <class T>
static inline void fill_raw(T* p, SizeType n, T v) {
  std::fill(p, p + n, v);
}

}  // namespace detail

template <class T>
class Conv2d final : public Layer<T> {
 public:
  using typename Layer<T>::ParamId;

  explicit Conv2d(const Conv2dOptions& opt)
      : opt_(opt), weight_id_(Layer<T>::allocate_param_id()) {
    validate_options_();

    weight_ = Tensor<T>(Shape{
        {opt_.out_channels, opt_.in_channels, opt_.kernel_h, opt_.kernel_w}});
    grad_w_ = Tensor<T>(weight_.shape());

    if (opt_.bias) {
      bias_ = Tensor<T>(Shape{{opt_.out_channels}});
      grad_b_ = Tensor<T>(bias_.shape());
      bias_id_ = Layer<T>::allocate_param_id();
    }

    init_seed_ = static_cast<uint32_t>(weight_id_);
    reset_parameters();
  }

  Conv2d(SizeType in_channels, SizeType out_channels, SizeType kernel_h,
         SizeType kernel_w, bool bias = true)
      : Conv2d(Conv2dOptions{in_channels, out_channels, kernel_h, kernel_w,
                             bias}) {}

  // Reinitialize parameters / grads / caches.
  // Default policy for Conv2d is He normal.
  void reset_parameters() override {
    switch (init_policy_) {
      case InitPolicy::kDefault:
      case InitPolicy::kHeNormal:
        init_params_he_(init_seed_);
        break;
      case InitPolicy::kXavierUniform:
        init_params_xavier_uniform_(init_seed_);
        break;
      case InitPolicy::kZeros:
        zero_params_();
        break;
    }
    zero_grads_();
    clear_cache_();
  }

  Shape output_shape(const Shape& in) const override {
    if (in.rank() != 4) {
      throw std::invalid_argument(
          "tinynn::Conv2d::output_shape: input rank must be 4 (NCHW)");
    }
    const SizeType N = in.dim(0);
    const SizeType Ci = in.dim(1);
    const SizeType H = in.dim(2);
    const SizeType W = in.dim(3);

    if (Ci != opt_.in_channels) {
      throw std::invalid_argument(
          "tinynn::Conv2d::output_shape: in_channels mismatch");
    }

    const SizeType Kh = opt_.kernel_h;
    const SizeType Kw = opt_.kernel_w;

    const SizeType Hpad = H + opt_.pad_h * 2;
    const SizeType Wpad = W + opt_.pad_w * 2;

    if (Hpad < Kh || Wpad < Kw) {
      throw std::invalid_argument(
          "tinynn::Conv2d::output_shape: kernel larger than padded input");
    }

    // stride=1, dilation=1 fixed
    const SizeType Ho = Hpad - Kh + 1;
    const SizeType Wo = Wpad - Kw + 1;
    return Shape{{N, opt_.out_channels, Ho, Wo}};
  }

  void forward(ConstTensorView<T> x, TensorView<T> y) override {
    const Shape out = output_shape(x.shape());
    if (y.shape() != out) {
      throw std::invalid_argument("tinynn::Conv2d::forward: y shape mismatch");
    }

    saved_x_ = Tensor<T>(x.shape());
    std::copy(x.data(), x.data() + x.size(), saved_x_.data());

    const SizeType N = x.shape().dim(0);
    const SizeType Ci = x.shape().dim(1);
    const SizeType H = x.shape().dim(2);
    const SizeType W = x.shape().dim(3);

    const SizeType Co = opt_.out_channels;
    const SizeType Kh = opt_.kernel_h;
    const SizeType Kw = opt_.kernel_w;

    const SizeType pad_h = opt_.pad_h;
    const SizeType pad_w = opt_.pad_w;

    const SizeType Ho = out.dim_unchecked(2);
    const SizeType Wo = out.dim_unchecked(3);

    detail::fill_raw(y.data(), y.size(), T{0});

    const T* xptr = saved_x_.data();
    const T* wptr = weight_.data();
    const T* bptr = opt_.bias ? bias_.data() : nullptr;
    T* yptr = y.data();

    for (SizeType n = 0; n < N; ++n) {
      for (SizeType co = 0; co < Co; ++co) {
        for (SizeType oh = 0; oh < Ho; ++oh) {
          for (SizeType ow = 0; ow < Wo; ++ow) {
            T acc = opt_.bias ? bptr[co] : T{0};

            const int ih_base = static_cast<int>(oh) - static_cast<int>(pad_h);
            const int iw_base = static_cast<int>(ow) - static_cast<int>(pad_w);

            for (SizeType ci = 0; ci < Ci; ++ci) {
              for (SizeType kh = 0; kh < Kh; ++kh) {
                for (SizeType kw = 0; kw < Kw; ++kw) {
                  const int ih = ih_base + static_cast<int>(kh);
                  const int iw = iw_base + static_cast<int>(kw);

                  if (0 <= ih && ih < static_cast<int>(H) &&
                      0 <= iw && iw < static_cast<int>(W)) {
                    const SizeType x_i =
                        detail::idx4(n, ci,
                                     static_cast<SizeType>(ih),
                                     static_cast<SizeType>(iw),
                                     Ci, H, W);
                    const SizeType w_i =
                        detail::idx4(co, ci, kh, kw, Ci, Kh, Kw);
                    acc += xptr[x_i] * wptr[w_i];
                  }
                }
              }
            }

            const SizeType y_i = detail::idx4(n, co, oh, ow, Co, Ho, Wo);
            yptr[y_i] = acc;
          }
        }
      }
    }
  }

  void backward(ConstTensorView<T> dy, TensorView<T> dx) override {
    if (saved_x_.empty()) {
      throw std::logic_error("tinynn::Conv2d::backward: called before forward");
    }

    const Shape out = output_shape(saved_x_.shape());
    if (dy.shape() != out) {
      throw std::invalid_argument(
          "tinynn::Conv2d::backward: dy shape mismatch");
    }
    if (dx.shape() != saved_x_.shape()) {
      throw std::invalid_argument(
          "tinynn::Conv2d::backward: dx shape mismatch");
    }

    const SizeType N = saved_x_.shape().dim(0);
    const SizeType Ci = saved_x_.shape().dim(1);
    const SizeType H = saved_x_.shape().dim(2);
    const SizeType W = saved_x_.shape().dim(3);

    const SizeType Co = opt_.out_channels;
    const SizeType Kh = opt_.kernel_h;
    const SizeType Kw = opt_.kernel_w;

    const SizeType pad_h = opt_.pad_h;
    const SizeType pad_w = opt_.pad_w;

    const SizeType Ho = out.dim_unchecked(2);
    const SizeType Wo = out.dim_unchecked(3);

    zero_grads_();
    detail::fill_raw(dx.data(), dx.size(), T{0});

    const T* xptr = saved_x_.data();
    const T* wptr = weight_.data();
    const T* gyptr = dy.data();

    T* gxptr = dx.data();
    T* gwptr = grad_w_.data();
    T* gbptr = opt_.bias ? grad_b_.data() : nullptr;

    for (SizeType n = 0; n < N; ++n) {
      for (SizeType co = 0; co < Co; ++co) {
        for (SizeType oh = 0; oh < Ho; ++oh) {
          for (SizeType ow = 0; ow < Wo; ++ow) {
            const SizeType gy_i = detail::idx4(n, co, oh, ow, Co, Ho, Wo);
            const T go = gyptr[gy_i];

            if (opt_.bias) {
              gbptr[co] += go;
            }

            const int ih_base = static_cast<int>(oh) - static_cast<int>(pad_h);
            const int iw_base = static_cast<int>(ow) - static_cast<int>(pad_w);

            for (SizeType ci = 0; ci < Ci; ++ci) {
              for (SizeType kh = 0; kh < Kh; ++kh) {
                for (SizeType kw = 0; kw < Kw; ++kw) {
                  const int ih = ih_base + static_cast<int>(kh);
                  const int iw = iw_base + static_cast<int>(kw);

                  if (0 <= ih && ih < static_cast<int>(H) &&
                      0 <= iw && iw < static_cast<int>(W)) {
                    const SizeType x_i =
                        detail::idx4(n, ci,
                                     static_cast<SizeType>(ih),
                                     static_cast<SizeType>(iw),
                                     Ci, H, W);
                    const SizeType w_i =
                        detail::idx4(co, ci, kh, kw, Ci, Kh, Kw);

                    // dW += x * dY
                    gwptr[w_i] += go * xptr[x_i];
                    // dX += W * dY
                    gxptr[x_i] += go * wptr[w_i];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  void collect_parameter_views(std::vector<ParameterView<T>>& out) override {
    out.push_back(ParameterView<T>{
        .param = weight_.view(),
        .grad  = grad_w_.view(),
        .kind  = ParamKind::kWeight,
        .id    = weight_id_,
    });

    if (opt_.bias) {
      out.push_back(ParameterView<T>{
          .param = bias_.view(),
          .grad  = grad_b_.view(),
          .kind  = ParamKind::kBias,
          .id    = bias_id_,
      });
    }
  }

  void set_init_policy(InitPolicy p) noexcept { init_policy_ = p; }
  void set_init_seed(uint32_t seed) noexcept { init_seed_ = seed; }
  [[nodiscard]] InitPolicy init_policy() const noexcept { return init_policy_; }
  [[nodiscard]] uint32_t init_seed() const noexcept { return init_seed_; }

 private:
  void validate_options_() const {
    if (opt_.in_channels == 0 || opt_.out_channels == 0) {
      throw std::invalid_argument("tinynn::Conv2d: channels must be > 0");
    }
    if (opt_.kernel_h == 0 || opt_.kernel_w == 0) {
      throw std::invalid_argument("tinynn::Conv2d: kernel must be > 0");
    }
    if (opt_.stride_h != 1 || opt_.stride_w != 1) {
      throw std::invalid_argument("tinynn::Conv2d: stride must be 1");
    }
    if (opt_.dilation_h != 1 || opt_.dilation_w != 1) {
      throw std::invalid_argument("tinynn::Conv2d: dilation must be 1");
    }
    if (opt_.groups != 1) {
      throw std::invalid_argument("tinynn::Conv2d: groups must be 1");
    }
  }

  void init_params_he_(uint32_t seed) {
    const double fan_in = static_cast<double>(opt_.in_channels) *
                          static_cast<double>(opt_.kernel_h) *
                          static_cast<double>(opt_.kernel_w);
    const double stddev = std::sqrt(2.0 / fan_in);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, stddev);

    for (SizeType i = 0; i < weight_.size(); ++i) {
      weight_[i] = static_cast<T>(nd(rng));
    }
    if (opt_.bias) {
      detail::fill_raw(bias_.data(), bias_.size(), T{0});
    }
  }

  void init_params_xavier_uniform_(uint32_t seed) {
    const double fan_in = static_cast<double>(opt_.in_channels) *
                          static_cast<double>(opt_.kernel_h) *
                          static_cast<double>(opt_.kernel_w);
    const double fan_out = static_cast<double>(opt_.out_channels) *
                           static_cast<double>(opt_.kernel_h) *
                           static_cast<double>(opt_.kernel_w);
    const double a = std::sqrt(6.0 / (fan_in + fan_out));

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> ud(-a, a);

    for (SizeType i = 0; i < weight_.size(); ++i) {
      weight_[i] = static_cast<T>(ud(rng));
    }
    if (opt_.bias) {
      detail::fill_raw(bias_.data(), bias_.size(), T{0});
    }
  }

  void zero_params_() {
    detail::fill_raw(weight_.data(), weight_.size(), T{0});
    if (opt_.bias) {
      detail::fill_raw(bias_.data(), bias_.size(), T{0});
    }
  }

  void zero_grads_() {
    detail::fill_raw(grad_w_.data(), grad_w_.size(), T{0});
    if (opt_.bias) {
      detail::fill_raw(grad_b_.data(), grad_b_.size(), T{0});
    }
  }

  void clear_cache_() {
    saved_x_ = Tensor<T>();
  }

 private:
  Conv2dOptions opt_;

  Tensor<T> weight_{};
  Tensor<T> grad_w_{};
  Tensor<T> bias_{};
  Tensor<T> grad_b_{};

  Tensor<T> saved_x_{};

  ParamId weight_id_{0};
  ParamId bias_id_{0};

  InitPolicy init_policy_ = InitPolicy::kDefault;
  uint32_t init_seed_ = 0;
};

}  // namespace tinynn
