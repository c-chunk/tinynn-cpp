#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <tinynn/nn/layer.h>
#include <tinynn/tensor/tensor.h>

namespace tinynn {

template <class T>
class BatchNorm2d final : public Layer<T> {
 public:
  using typename Layer<T>::ParamId;

  explicit BatchNorm2d(
      SizeType num_channels,
      T eps = static_cast<T>(1e-5),
      T momentum = static_cast<T>(0.1))
      : C_(num_channels),
        eps_(eps),
        momentum_(momentum),
        gamma_({C_}, static_cast<T>(1)),
        beta_({C_}, static_cast<T>(0)),
        grad_gamma_({C_}),
        grad_beta_({C_}),
        running_mean_({C_}, static_cast<T>(0)),
        running_var_({C_}, static_cast<T>(1)),
        inv_std_({C_}),
        gamma_id_(Layer<T>::allocate_param_id()),
        beta_id_(Layer<T>::allocate_param_id()),
        running_mean_id_(Layer<T>::allocate_param_id()),
        running_var_id_(Layer<T>::allocate_param_id()) {
    if (C_ == 0) {
      throw std::invalid_argument("BatchNorm2d: num_channels must be > 0");
    }
    scratch_sum_.resize(static_cast<size_t>(C_));
    scratch_sqsum_.resize(static_cast<size_t>(C_));
    scratch_mean_.resize(static_cast<size_t>(C_));
    scratch_var_.resize(static_cast<size_t>(C_));

    reset_parameters();
  }

  void reset_parameters() override {
    for (SizeType c = 0; c < C_; ++c) {
      gamma_[c] = static_cast<T>(1);
      beta_[c] = static_cast<T>(0);
      grad_gamma_[c] = static_cast<T>(0);
      grad_beta_[c] = static_cast<T>(0);
      running_mean_[c] = static_cast<T>(0);
      running_var_[c] = static_cast<T>(1);
      inv_std_[c] = static_cast<T>(1);
    }
    clear_cache_();
  }

  Shape output_shape(const Shape& in) const override {
    if (in.rank() != 4) {
      throw std::invalid_argument("BatchNorm2d: input must be NCHW (rank 4)");
    }
    if (in.dim(1) != C_) {
      throw std::invalid_argument("BatchNorm2d: channel mismatch");
    }
    return in;
  }

  void forward(ConstTensorView<T> x, TensorView<T> y) override {
    if (x.shape() != y.shape()) {
      throw std::invalid_argument("BatchNorm2d::forward: shape mismatch");
    }
    if (x.shape().rank() != 4) {
      throw std::invalid_argument("BatchNorm2d::forward: rank must be 4");
    }

    const SizeType N = x.shape().dim(0);
    const SizeType C = x.shape().dim(1);
    const SizeType H = x.shape().dim(2);
    const SizeType W = x.shape().dim(3);

    if (C != C_) {
      throw std::invalid_argument("BatchNorm2d::forward: channel mismatch");
    }

    const SizeType M = N * H * W;
    if (M == 0) {
      throw std::invalid_argument("BatchNorm2d::forward: N*H*W must be > 0");
    }

    x_hat_ = Tensor<T>(x.shape());

    if (this->is_training()) {
      compute_train_stats_and_forward_(x, y, N, C, H, W);
    } else {
      compute_eval_forward_(x, y, N, C, H, W);
    }
  }

  void backward(ConstTensorView<T> dy, TensorView<T> dx) override {
    if (dy.shape() != dx.shape()) {
      throw std::invalid_argument("BatchNorm2d::backward: shape mismatch");
    }
    if (dy.shape().rank() != 4) {
      throw std::invalid_argument("BatchNorm2d::backward: rank must be 4");
    }
    if (x_hat_.empty()) {
      throw std::logic_error(
          "BatchNorm2d::backward: missing cache (call forward first)");
    }

    const SizeType N = dy.shape().dim(0);
    const SizeType C = dy.shape().dim(1);
    const SizeType H = dy.shape().dim(2);
    const SizeType W = dy.shape().dim(3);
    const SizeType M = N * H * W;

    if (C != C_) {
      throw std::invalid_argument("BatchNorm2d::backward: channel mismatch");
    }
    if (M == 0) {
      throw std::invalid_argument("BatchNorm2d::backward: N*H*W must be > 0");
    }

    std::fill(grad_gamma_.data(),
              grad_gamma_.data() + grad_gamma_.size(),
              static_cast<T>(0));
    std::fill(grad_beta_.data(),
              grad_beta_.data() + grad_beta_.size(),
              static_cast<T>(0));

    // pass 1:
    //   dbeta  = sum(dy)
    //   dgamma = sum(dy * xhat)
    {
      const T* dyp = dy.data();
      const T* xhp = x_hat_.data();

      SizeType idx = 0;
      for (SizeType n = 0; n < N; ++n) {
        for (SizeType c = 0; c < C; ++c) {
          const SizeType hw = H * W;
          T sum_dy = static_cast<T>(0);
          T sum_dy_xhat = static_cast<T>(0);

          for (SizeType k = 0; k < hw; ++k, ++idx) {
            const T g = dyp[idx];
            const T xh = xhp[idx];
            sum_dy += g;
            sum_dy_xhat += g * xh;
          }

          grad_beta_[c] += sum_dy;
          grad_gamma_[c] += sum_dy_xhat;
        }
      }
    }

    if (this->is_training()) {
      // pass 2:
      // dx = (1/M) * invstd * gamma * (M*dy - sum(dy) - xhat*sum(dy*xhat))
      const T invM = static_cast<T>(1) / static_cast<T>(M);

      const T* dyp = dy.data();
      const T* xhp = x_hat_.data();
      T* dxp = dx.data();

      SizeType idx = 0;
      for (SizeType n = 0; n < N; ++n) {
        for (SizeType c = 0; c < C; ++c) {
          const T gamma = gamma_[c];
          const T inv_std = inv_std_[c];
          const T sum_dy = grad_beta_[c];
          const T sum_dy_xhat = grad_gamma_[c];

          const SizeType hw = H * W;
          for (SizeType k = 0; k < hw; ++k, ++idx) {
            const T g = dyp[idx];
            const T xh = xhp[idx];
            const T term =
                static_cast<T>(M) * g - sum_dy - xh * sum_dy_xhat;
            dxp[idx] = gamma * inv_std * invM * term;
          }
        }
      }
    } else {
      const T* dyp = dy.data();
      T* dxp = dx.data();

      SizeType idx = 0;
      for (SizeType n = 0; n < N; ++n) {
        for (SizeType c = 0; c < C; ++c) {
          const T scale = gamma_[c] * inv_std_[c];
          const SizeType hw = H * W;
          for (SizeType k = 0; k < hw; ++k, ++idx) {
            dxp[idx] = dyp[idx] * scale;
          }
        }
      }
    }

    clear_cache_();
  }

  void collect_parameter_views(std::vector<ParameterView<T>>& out) override {
    out.push_back(ParameterView<T>{
        gamma_.view(),
        grad_gamma_.view(),
        ParamKind::kWeight,
        gamma_id_});

    out.push_back(ParameterView<T>{
        beta_.view(),
        grad_beta_.view(),
        ParamKind::kBias,
        beta_id_});
  }

  void collect_buffer_views(std::vector<BufferView<T>>& out) override {
    out.push_back(BufferView<T>{running_mean_.view(), running_mean_id_});
    out.push_back(BufferView<T>{running_var_.view(), running_var_id_});
  }

  [[nodiscard]] const Tensor<T>& running_mean() const noexcept {
    return running_mean_;
  }

  [[nodiscard]] const Tensor<T>& running_var() const noexcept {
    return running_var_;
  }

  [[nodiscard]] T eps() const noexcept { return eps_; }

  [[nodiscard]] T momentum() const noexcept { return momentum_; }

 private:
  void compute_train_stats_and_forward_(
      ConstTensorView<T> x, TensorView<T> y,
      SizeType N, SizeType C, SizeType H, SizeType W) {
    const SizeType hw = H * W;
    const SizeType M = N * hw;
    const T invM = static_cast<T>(1) / static_cast<T>(M);

    std::fill(scratch_sum_.begin(), scratch_sum_.end(), static_cast<T>(0));
    std::fill(scratch_sqsum_.begin(), scratch_sqsum_.end(), static_cast<T>(0));

    const T* xp = x.data();

    // pass A: sum / sqsum
    SizeType idx = 0;
    for (SizeType n = 0; n < N; ++n) {
      for (SizeType c = 0; c < C; ++c) {
        T s = static_cast<T>(0);
        T ss = static_cast<T>(0);

        for (SizeType k = 0; k < hw; ++k, ++idx) {
          const T v = xp[idx];
          s += v;
          ss += v * v;
        }

        scratch_sum_[c] += s;
        scratch_sqsum_[c] += ss;
      }
    }

    // mean / var / invstd + running update
    for (SizeType c = 0; c < C; ++c) {
      const T mean = scratch_sum_[c] * invM;
      T var = scratch_sqsum_[c] * invM - mean * mean;
      if (var < static_cast<T>(0)) {
        var = static_cast<T>(0);
      }

      scratch_mean_[c] = mean;
      scratch_var_[c] = var;
      inv_std_[c] = static_cast<T>(1) / std::sqrt(var + eps_);

      running_mean_[c] =
          (static_cast<T>(1) - momentum_) * running_mean_[c]
          + momentum_ * mean;

      running_var_[c] =
          (static_cast<T>(1) - momentum_) * running_var_[c]
          + momentum_ * var;
    }

    // pass B: xhat + y
    T* xhp = x_hat_.data();
    T* yp = y.data();
    idx = 0;

    for (SizeType n = 0; n < N; ++n) {
      for (SizeType c = 0; c < C; ++c) {
        const T mean = scratch_mean_[c];
        const T inv_std = inv_std_[c];
        const T gamma = gamma_[c];
        const T beta = beta_[c];

        for (SizeType k = 0; k < hw; ++k, ++idx) {
          const T xh = (xp[idx] - mean) * inv_std;
          xhp[idx] = xh;
          yp[idx] = gamma * xh + beta;
        }
      }
    }
  }

  void compute_eval_forward_(
      ConstTensorView<T> x, TensorView<T> y,
      SizeType N, SizeType C, SizeType H, SizeType W) {
    for (SizeType c = 0; c < C; ++c) {
      inv_std_[c] =
          static_cast<T>(1) / std::sqrt(running_var_[c] + eps_);
    }

    const SizeType hw = H * W;
    const T* xp = x.data();
    T* xhp = x_hat_.data();
    T* yp = y.data();

    SizeType idx = 0;
    for (SizeType n = 0; n < N; ++n) {
      for (SizeType c = 0; c < C; ++c) {
        const T mean = running_mean_[c];
        const T inv_std = inv_std_[c];
        const T gamma = gamma_[c];
        const T beta = beta_[c];

        for (SizeType k = 0; k < hw; ++k, ++idx) {
          const T xh = (xp[idx] - mean) * inv_std;
          xhp[idx] = xh;
          yp[idx] = gamma * xh + beta;
        }
      }
    }
  }

  void clear_cache_() {
    x_hat_ = Tensor<T>();
  }

 private:
  SizeType C_;
  T eps_;
  T momentum_;

  Tensor<T> gamma_;
  Tensor<T> beta_;
  Tensor<T> grad_gamma_;
  Tensor<T> grad_beta_;

  Tensor<T> running_mean_;
  Tensor<T> running_var_;

  Tensor<T> inv_std_;   // [C]
  Tensor<T> x_hat_;     // [N,C,H,W]

  // reuse scratch
  std::vector<T> scratch_sum_;    // [C]
  std::vector<T> scratch_sqsum_;  // [C]
  std::vector<T> scratch_mean_;   // [C]
  std::vector<T> scratch_var_;    // [C]

  ParamId gamma_id_;
  ParamId beta_id_;

  ParamId running_mean_id_;
  ParamId running_var_id_;
};

}  // namespace tinynn
