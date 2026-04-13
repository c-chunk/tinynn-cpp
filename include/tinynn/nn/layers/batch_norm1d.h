#pragma once

#include <stdexcept>
#include <vector>

#include <tinynn/nn/layer.h>
#include <tinynn/tensor/tensor.h>

namespace tinynn {

template <class T>
class BatchNorm1d final : public Layer<T> {
 public:
  using typename Layer<T>::ParamId;

  explicit BatchNorm1d(
      SizeType num_features,
      T eps = static_cast<T>(1e-5),
      T momentum = static_cast<T>(0.1))
      : num_features_(num_features),
        eps_(eps),
        momentum_(momentum),
        gamma_({num_features}, static_cast<T>(1)),
        beta_({num_features}, static_cast<T>(0)),
        grad_gamma_({num_features}),
        grad_beta_({num_features}),
        running_mean_({num_features}, static_cast<T>(0)),
        running_var_({num_features}, static_cast<T>(1)),
        gamma_id_(Layer<T>::allocate_param_id()),
        beta_id_(Layer<T>::allocate_param_id()),
        running_mean_id_(Layer<T>::allocate_param_id()),
        running_var_id_(Layer<T>::allocate_param_id()) {
    if (num_features_ == 0) {
      throw std::invalid_argument("BatchNorm1d: num_features must be > 0");
    }

    reset_parameters();
  }

  void reset_parameters() override {
    for (SizeType c = 0; c < num_features_; ++c) {
      gamma_[c] = static_cast<T>(1);
      beta_[c] = static_cast<T>(0);
      grad_gamma_[c] = static_cast<T>(0);
      grad_beta_[c] = static_cast<T>(0);
      running_mean_[c] = static_cast<T>(0);
      running_var_[c] = static_cast<T>(1);
    }
    clear_cache_();
  }

  Shape output_shape(const Shape& input_shape) const override {
    if (input_shape.rank() != 2) {
      throw std::invalid_argument("BatchNorm1d: input rank must be 2");
    }
    if (input_shape.cols() != num_features_) {
      throw std::invalid_argument("BatchNorm1d: feature size mismatch");
    }
    return input_shape;
  }

  void forward(ConstTensorView<T> x, TensorView<T> y) override {
    if (x.shape() != y.shape()) {
      throw std::invalid_argument("BatchNorm1d::forward: shape mismatch");
    }
    if (x.shape().rank() != 2) {
      throw std::invalid_argument("BatchNorm1d::forward: rank must be 2");
    }

    const SizeType B = x.row_count();
    const SizeType C = x.col_count();

    if (this->is_training()) {
      compute_batch_stats_(x, B, C);
      update_running_stats_(C);
      normalize_forward_from_xhat_(y, B, C);
    } else {
      compute_eval_cache_(x, B, C);
      normalize_forward_from_xhat_(y, B, C);
    }
  }

  void backward(ConstTensorView<T> dy, TensorView<T> dx) override {
    if (dy.shape() != dx.shape()) {
      throw std::invalid_argument("BatchNorm1d::backward: shape mismatch");
    }

    const SizeType B = dy.row_count();
    const SizeType C = dy.col_count();

    if (x_hat_.empty() || inv_std_.empty()) {
      throw std::invalid_argument(
          "BatchNorm1d::backward: missing cache (call forward first)");
    }

    for (SizeType c = 0; c < C; ++c) {
      grad_gamma_[c] = static_cast<T>(0);
      grad_beta_[c] = static_cast<T>(0);
    }

    for (SizeType b = 0; b < B; ++b) {
      for (SizeType c = 0; c < C; ++c) {
        grad_beta_[c] += dy(b, c);
        grad_gamma_[c] += dy(b, c) * x_hat_(b, c);
      }
    }

    if (this->is_training()) {
      for (SizeType b = 0; b < B; ++b) {
        for (SizeType c = 0; c < C; ++c) {
          const T inv_std = inv_std_[c];
          const T gamma = gamma_[c];

          const T term =
              static_cast<T>(B) * dy(b, c)
              - grad_beta_[c]
              - x_hat_(b, c) * grad_gamma_[c];

          dx(b, c) =
              (gamma * inv_std / static_cast<T>(B)) * term;
        }
      }
    } else {
      for (SizeType b = 0; b < B; ++b) {
        for (SizeType c = 0; c < C; ++c) {
          dx(b, c) = dy(b, c) * gamma_[c] * inv_std_[c];
        }
      }
    }

    clear_cache_();
  }

  void collect_parameter_views(
      std::vector<ParameterView<T>>& out) override {
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
  void compute_batch_stats_(ConstTensorView<T> x, SizeType B, SizeType C) {
    batch_mean_ = Tensor<T>({C});
    batch_var_ = Tensor<T>({C});
    inv_std_ = Tensor<T>({C});
    x_hat_ = Tensor<T>({B, C});

    for (SizeType c = 0; c < C; ++c) {
      T mean = static_cast<T>(0);
      for (SizeType b = 0; b < B; ++b) {
        mean += x(b, c);
      }
      mean /= static_cast<T>(B);
      batch_mean_[c] = mean;
    }

    for (SizeType c = 0; c < C; ++c) {
      T var = static_cast<T>(0);
      for (SizeType b = 0; b < B; ++b) {
        const T diff = x(b, c) - batch_mean_[c];
        var += diff * diff;
      }
      var /= static_cast<T>(B);
      batch_var_[c] = var;
      inv_std_[c] = static_cast<T>(1) / std::sqrt(var + eps_);
    }

    for (SizeType b = 0; b < B; ++b) {
      for (SizeType c = 0; c < C; ++c) {
        x_hat_(b, c) = (x(b, c) - batch_mean_[c]) * inv_std_[c];
      }
    }
  }

  void compute_eval_cache_(ConstTensorView<T> x, SizeType B, SizeType C) {
    inv_std_ = Tensor<T>({C});
    x_hat_ = Tensor<T>({B, C});

    for (SizeType c = 0; c < C; ++c) {
      inv_std_[c] =
          static_cast<T>(1) / std::sqrt(running_var_[c] + eps_);
    }

    for (SizeType b = 0; b < B; ++b) {
      for (SizeType c = 0; c < C; ++c) {
        x_hat_(b, c) =
            (x(b, c) - running_mean_[c]) * inv_std_[c];
      }
    }
  }

  void update_running_stats_(SizeType C) {
    for (SizeType c = 0; c < C; ++c) {
      running_mean_[c] =
          (static_cast<T>(1) - momentum_) * running_mean_[c]
          + momentum_ * batch_mean_[c];

      running_var_[c] =
          (static_cast<T>(1) - momentum_) * running_var_[c]
          + momentum_ * batch_var_[c];
    }
  }

  void normalize_forward_from_xhat_(TensorView<T> y, SizeType B, SizeType C) {
    for (SizeType b = 0; b < B; ++b) {
      for (SizeType c = 0; c < C; ++c) {
        y(b, c) = gamma_[c] * x_hat_(b, c) + beta_[c];
      }
    }
  }

  void clear_cache_() {
    x_hat_ = Tensor<T>();
    batch_mean_ = Tensor<T>();
    batch_var_ = Tensor<T>();
    inv_std_ = Tensor<T>();
  }

 private:
  SizeType num_features_;
  T eps_;
  T momentum_;

  Tensor<T> gamma_;
  Tensor<T> beta_;
  Tensor<T> grad_gamma_;
  Tensor<T> grad_beta_;

  Tensor<T> running_mean_;
  Tensor<T> running_var_;

  Tensor<T> batch_mean_;
  Tensor<T> batch_var_;
  Tensor<T> inv_std_;
  Tensor<T> x_hat_;

  ParamId gamma_id_;
  ParamId beta_id_;

  ParamId running_mean_id_;
  ParamId running_var_id_;
};

}  // namespace tinynn
