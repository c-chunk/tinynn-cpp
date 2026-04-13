#pragma once

#include <cassert>
#include <random>
#include <stdexcept>
#include <vector>

#include <tinynn/nn/init_policy.h>
#include <tinynn/nn/layer.h>
#include <tinynn/nn/parameter_view.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

// Affine layer: y = x * W + b
// x: [batch, in_dim], W: [in_dim, out_dim], b: [out_dim], y: [batch, out_dim]
template <class T>
class Affine final : public Layer<T> {
 public:
  using typename Layer<T>::ParamId;

  Affine(SizeType in_dim, SizeType out_dim, bool with_bias = true)
      : in_dim_(in_dim),
        out_dim_(out_dim),
        with_bias_(with_bias),
        W_(Shape{in_dim_, out_dim_}),
        dW_(Shape{in_dim_, out_dim_}),
        wid_(Layer<T>::allocate_param_id()) {
    if (in_dim_ == 0 || out_dim_ == 0) {
      throw std::invalid_argument("tinynn::Affine: in_dim/out_dim must be > 0");
    }

    if (with_bias_) {
      b_ = Tensor<T>(Shape{out_dim_});
      db_ = Tensor<T>(Shape{out_dim_});
      bid_ = Layer<T>::allocate_param_id();
    }

    init_seed_ = static_cast<uint32_t>(wid_);
    reset_parameters();
  }

  SizeType in_dim() const noexcept { return in_dim_; }
  SizeType out_dim() const noexcept { return out_dim_; }
  bool with_bias() const noexcept { return with_bias_; }

  // Reinitialize parameters / grads / caches.
  // Default policy for Affine is He normal + zero bias.
  void reset_parameters() override {
    switch (init_policy_) {
      case InitPolicy::kDefault:
      case InitPolicy::kHeNormal:
        init_he_normal(init_seed_);
        break;
      case InitPolicy::kXavierUniform:
        init_xavier_uniform(init_seed_);
        break;
      case InitPolicy::kZeros:
        zero_params_and_grads();
        clear_cache();
        return;
    }
    zero_grads();
    clear_cache();
  }

  // Optional: set parameters from outside (e.g., initializer)
  TensorView<T> W() noexcept { return W_.view(); }
  ConstTensorView<T> W() const noexcept { return W_.view(); }

  TensorView<T> b() {
    assert(with_bias_);
    if (!with_bias_) {
      throw std::logic_error("tinynn::Affine::b: bias is disabled");
    }
    return b_.view();
  }

  ConstTensorView<T> b() const {
    assert(with_bias_);
    if (!with_bias_) {
      throw std::logic_error("tinynn::Affine::b: bias is disabled");
    }
    return b_.view();
  }

  Shape output_shape(const Shape& input_shape) const override {
    if (input_shape.rank() != 2) {
      throw std::invalid_argument("tinynn::Affine: input must be rank-2");
    }
    if (input_shape.dim_unchecked(1) != in_dim_) {
      throw std::invalid_argument("tinynn::Affine: input col_count != in_dim");
    }
    return Shape{input_shape.dim_unchecked(0), out_dim_};  // [batch, out_dim]
  }

  void forward(ConstTensorView<T> x, TensorView<T> y) override {
    // Debug-only invariants / fast fail
    assert(x.shape().rank() == 2);
    assert(y.shape().rank() == 2);
    assert(x.col_count() == in_dim_);
    assert(y.col_count() == out_dim_);
    assert(y.row_count() == x.row_count());

    // Parameter shapes
    assert(W_.view().shape().rank() == 2);
    assert(W_.view().row_count() == in_dim_);
    assert(W_.view().col_count() == out_dim_);
    if (with_bias_) {
      assert(b_.view().shape().rank() == 1);
      assert(b_.view().size() == out_dim_);
    }

    ensure_forward_shapes(x, y);

    // Cache input view for backward.
    // Contract: x's underlying storage must remain valid until backward().
    last_x_ = x;

    const SizeType batch = x.row_count();

    // y = xW (+b)
    for (SizeType r = 0; r < batch; ++r) {
      auto xr = x.row_span(r);  // [in_dim]
      auto yr = y.row_span(r);  // [out_dim]

      for (SizeType c = 0; c < out_dim_; ++c) {
        T acc = static_cast<T>(0);
        for (SizeType i = 0; i < in_dim_; ++i) {
          acc += xr[i] * W_(i, c);
        }
        if (with_bias_) {
          acc += b_[c];
        }
        yr[c] = acc;
      }
    }
  }

  void backward(ConstTensorView<T> dy, TensorView<T> dx) override {
    // Debug-only invariants / fast fail
    assert(dy.shape().rank() == 2);
    assert(dx.shape().rank() == 2);
    assert(dy.col_count() == out_dim_);
    assert(dx.col_count() == in_dim_);
    assert(dx.row_count() == dy.row_count());

    // Must have cached input from forward
    assert(last_x_.data() != nullptr);
    if (last_x_.data() != nullptr) {
      assert(last_x_.shape().rank() == 2);
      assert(last_x_.col_count() == in_dim_);
      assert(last_x_.row_count() == dy.row_count());
    }

    ensure_backward_shapes(dy, dx);
    ensure_cached_input();

    // Cached input must match current batch (defensive)
    if (last_x_.row_count() != dy.row_count()) {
      throw std::invalid_argument(
          "tinynn::Affine::backward: batch mismatch with cached input");
    }

    zero_grads();

    const SizeType batch = dy.row_count();

    // dx = dy * W^T
    for (SizeType r = 0; r < batch; ++r) {
      auto dyr = dy.row_span(r);  // [out_dim]
      auto dxr = dx.row_span(r);  // [in_dim]

      for (SizeType i = 0; i < in_dim_; ++i) {
        T acc = static_cast<T>(0);
        for (SizeType c = 0; c < out_dim_; ++c) {
          acc += dyr[c] * W_(i, c);
        }
        dxr[i] = acc;
      }
    }

    // dW = x^T * dy
    // db = sum_rows(dy)
    for (SizeType r = 0; r < batch; ++r) {
      auto xr = last_x_.row_span(r);  // [in_dim]
      auto dyr = dy.row_span(r);      // [out_dim]

      for (SizeType i = 0; i < in_dim_; ++i) {
        const T xi = xr[i];
        for (SizeType c = 0; c < out_dim_; ++c) {
          dW_(i, c) += xi * dyr[c];
        }
      }

      if (with_bias_) {
        for (SizeType c = 0; c < out_dim_; ++c) {
          db_[c] += dyr[c];
        }
      }
    }
  }

  void collect_parameter_views(std::vector<ParameterView<T>>& out) override {
    out.push_back(ParameterView<T>{
        /*param=*/W_.view(),
        /*grad =*/dW_.view(),
        /*kind=*/ParamKind::kWeight,
        /*id  =*/wid_,
    });

    if (with_bias_) {
      out.push_back(ParameterView<T>{
          /*param=*/b_.view(),
          /*grad =*/db_.view(),
          /*kind=*/ParamKind::kBias,
          /*id  =*/bid_,
      });
    }
  }

  // Optional: clear cached input (e.g., after one training step)
  void clear_cache() noexcept { last_x_ = ConstTensorView<T>(); }

  // Xavier/Glorot uniform: U[-a, a], a = sqrt(6 / (fan_in + fan_out))
  void init_xavier_uniform(uint32_t seed = 123) {
    const T fan_in = static_cast<T>(in_dim_);
    const T fan_out = static_cast<T>(out_dim_);
    const T a =
        static_cast<T>(std::sqrt(6.0 / static_cast<double>(fan_in + fan_out)));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(
        -static_cast<double>(a), static_cast<double>(a));

    for (SizeType i = 0; i < W_.size(); ++i) {
      W_[i] = static_cast<T>(dist(rng));
    }

    if (with_bias_) {
      for (SizeType i = 0; i < b_.size(); ++i) {
        b_[i] = static_cast<T>(0);
      }
    }
  }

  // He/Kaiming normal: N(0, std), std = sqrt(2 / fan_in)  (for ReLU)
  void init_he_normal(uint32_t seed = 123) {
    const T fan_in = static_cast<T>(in_dim_);
    const T stddev =
        static_cast<T>(std::sqrt(2.0 / static_cast<double>(fan_in)));

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, static_cast<double>(stddev));

    for (SizeType i = 0; i < W_.size(); ++i) {
      W_[i] = static_cast<T>(dist(rng));
    }

    if (with_bias_) {
      for (SizeType i = 0; i < b_.size(); ++i) {
        b_[i] = static_cast<T>(0);
      }
    }
  }

  void set_init_policy(InitPolicy p) noexcept { init_policy_ = p; }
  void set_init_seed(uint32_t seed) noexcept { init_seed_ = seed; }
  [[nodiscard]] InitPolicy init_policy() const noexcept { return init_policy_; }
  [[nodiscard]] uint32_t init_seed() const noexcept { return init_seed_; }

 private:
  void ensure_forward_shapes(ConstTensorView<T> x, TensorView<T> y) const {
    if (x.shape().rank() != 2 || y.shape().rank() != 2) {
      throw std::invalid_argument("tinynn::Affine::forward: x/y must be rank-2");
    }
    if (x.col_count() != in_dim_) {
      throw std::invalid_argument("tinynn::Affine::forward: x col_count != in_dim");
    }
    if (y.col_count() != out_dim_) {
      throw std::invalid_argument("tinynn::Affine::forward: y col_count != out_dim");
    }
    if (y.row_count() != x.row_count()) {
      throw std::invalid_argument("tinynn::Affine::forward: y rows != x rows");
    }
  }

  void ensure_backward_shapes(ConstTensorView<T> dy, TensorView<T> dx) const {
    if (dy.shape().rank() != 2 || dx.shape().rank() != 2) {
      throw std::invalid_argument("tinynn::Affine::backward: dy/dx must be rank-2");
    }
    if (dy.col_count() != out_dim_) {
      throw std::invalid_argument("tinynn::Affine::backward: dy col_count != out_dim");
    }
    if (dx.col_count() != in_dim_) {
      throw std::invalid_argument("tinynn::Affine::backward: dx col_count != in_dim");
    }
    if (dx.row_count() != dy.row_count()) {
      throw std::invalid_argument("tinynn::Affine::backward: dx rows != dy rows");
    }
  }

  void ensure_cached_input() const {
    if (last_x_.data() == nullptr) {
      throw std::invalid_argument(
          "tinynn::Affine::backward: missing cached input (call forward first)");
    }
    if (last_x_.shape().rank() != 2 || last_x_.col_count() != in_dim_) {
      throw std::invalid_argument(
          "tinynn::Affine::backward: cached input shape mismatch");
    }
  }

  void zero_grads() {
    for (SizeType i = 0; i < dW_.size(); ++i) {
      dW_[i] = static_cast<T>(0);
    }
    if (with_bias_) {
      for (SizeType i = 0; i < db_.size(); ++i) {
        db_[i] = static_cast<T>(0);
      }
    }
  }

  void zero_params_and_grads() {
    for (SizeType i = 0; i < W_.size(); ++i) {
      W_[i] = static_cast<T>(0);
    }
    if (with_bias_) {
      for (SizeType i = 0; i < b_.size(); ++i) {
        b_[i] = static_cast<T>(0);
      }
    }
    zero_grads();
  }

  // Optional if you decide to average gradients by batch.
  void scale_grads(T scale) {
    for (SizeType i = 0; i < dW_.size(); ++i) {
      dW_[i] *= scale;
    }
    if (with_bias_) {
      for (SizeType i = 0; i < db_.size(); ++i) {
        db_[i] *= scale;
      }
    }
  }

 private:
  SizeType in_dim_ = 0;
  SizeType out_dim_ = 0;
  bool with_bias_ = true;

  Tensor<T> W_;
  Tensor<T> dW_;
  Tensor<T> b_;
  Tensor<T> db_;

  // Cached input view for backward (non-owning).
  ConstTensorView<T> last_x_{};

  ParamId wid_ = 0;
  ParamId bid_ = 0;

  InitPolicy init_policy_ = InitPolicy::kDefault;
  uint32_t init_seed_ = 0;
};

}  // namespace tinynn
