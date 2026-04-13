#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include <tinynn/nn/layer.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

// Dropout2d (channel-wise / spatial dropout for NCHW)
// - Input/Output: rank-4 tensor [N, C, H, W]
// - Train:
//     for each (n, c), either keep the whole HxW plane or drop it
//     inverted dropout scaling: y = x * mask * (1 / (1-p))
// - Eval:
//     identity
template <class T>
class Dropout2d final : public Layer<T> {
 public:
  explicit Dropout2d(T drop_prob, std::uint64_t seed = 0xD0A2D0A2ULL)
      : drop_prob_(drop_prob),
        keep_prob_(static_cast<T>(1) - drop_prob),
        scale_(static_cast<T>(1)),
        rng_(seed),
        bernoulli_(static_cast<double>(keep_prob_)) {
    validate_prob_();
    if (drop_prob_ < static_cast<T>(1)) {
      scale_ = static_cast<T>(1) / keep_prob_;
    }
  }

  [[nodiscard]] T drop_prob() const noexcept { return drop_prob_; }
  [[nodiscard]] T keep_prob() const noexcept { return keep_prob_; }

  void set_seed(std::uint64_t seed) {
    rng_.seed(seed);
  }

  Shape output_shape(const Shape& in) const override {
    if (in.rank() != 4) {
      throw std::invalid_argument(
          "tinynn::Dropout2d::output_shape: input must be rank-4 (NCHW)");
    }
    return in;
  }

  void forward(ConstTensorView<T> x, TensorView<T> y) override {
    if (x.shape() != y.shape()) {
      throw std::invalid_argument(
          "tinynn::Dropout2d::forward: shape mismatch");
    }
    if (x.shape().rank() != 4) {
      throw std::invalid_argument(
          "tinynn::Dropout2d::forward: input must be rank-4 (NCHW)");
    }

    const SizeType N = x.shape().dim(0);
    const SizeType C = x.shape().dim(1);
    const SizeType H = x.shape().dim(2);
    const SizeType W = x.shape().dim(3);
    const SizeType HW = H * W;

    if (!this->is_training()) {
      std::copy(x.data(), x.data() + x.size(), y.data());
      clear_mask_();
      return;
    }

    if (drop_prob_ == static_cast<T>(0)) {
      std::copy(x.data(), x.data() + x.size(), y.data());
      mask_.assign(static_cast<size_t>(N * C), 1u);
      last_n_ = N;
      last_c_ = C;
      return;
    }

    if (drop_prob_ == static_cast<T>(1)) {
      std::fill(y.data(), y.data() + y.size(), static_cast<T>(0));
      mask_.assign(static_cast<size_t>(N * C), 0u);
      last_n_ = N;
      last_c_ = C;
      return;
    }

    mask_.resize(static_cast<size_t>(N * C));
    last_n_ = N;
    last_c_ = C;

    const T* xp = x.data();
    T* yp = y.data();

    SizeType idx = 0;
    SizeType mask_idx = 0;
    for (SizeType n = 0; n < N; ++n) {
      for (SizeType c = 0; c < C; ++c, ++mask_idx) {
        const std::uint8_t keep =
            bernoulli_(rng_) ? static_cast<std::uint8_t>(1)
                             : static_cast<std::uint8_t>(0);
        mask_[static_cast<size_t>(mask_idx)] = keep;

        if (keep) {
          for (SizeType k = 0; k < HW; ++k, ++idx) {
            yp[idx] = xp[idx] * scale_;
          }
        } else {
          for (SizeType k = 0; k < HW; ++k, ++idx) {
            yp[idx] = static_cast<T>(0);
          }
        }
      }
    }
  }

  void backward(ConstTensorView<T> dy, TensorView<T> dx) override {
    if (dy.shape() != dx.shape()) {
      throw std::invalid_argument(
          "tinynn::Dropout2d::backward: shape mismatch");
    }
    if (dy.shape().rank() != 4) {
      throw std::invalid_argument(
          "tinynn::Dropout2d::backward: input must be rank-4 (NCHW)");
    }

    const SizeType N = dy.shape().dim(0);
    const SizeType C = dy.shape().dim(1);
    const SizeType H = dy.shape().dim(2);
    const SizeType W = dy.shape().dim(3);
    const SizeType HW = H * W;

    if (!this->is_training()) {
      std::copy(dy.data(), dy.data() + dy.size(), dx.data());
      return;
    }

    if (last_n_ != N || last_c_ != C ||
        mask_.size() != static_cast<size_t>(N * C)) {
      throw std::invalid_argument(
          "tinynn::Dropout2d::backward: missing or invalid mask "
          "(call forward first)");
    }

    const T* dyp = dy.data();
    T* dxp = dx.data();

    SizeType idx = 0;
    SizeType mask_idx = 0;
    for (SizeType n = 0; n < N; ++n) {
      for (SizeType c = 0; c < C; ++c, ++mask_idx) {
        const std::uint8_t keep = mask_[static_cast<size_t>(mask_idx)];

        if (keep) {
          for (SizeType k = 0; k < HW; ++k, ++idx) {
            dxp[idx] = dyp[idx] * scale_;
          }
        } else {
          for (SizeType k = 0; k < HW; ++k, ++idx) {
            dxp[idx] = static_cast<T>(0);
          }
        }
      }
    }

    clear_mask_();
  }

 private:
  void validate_prob_() const {
    if (drop_prob_ < static_cast<T>(0) || drop_prob_ > static_cast<T>(1)) {
      throw std::invalid_argument(
          "tinynn::Dropout2d: drop_prob must be in [0, 1]");
    }
  }

  void clear_mask_() noexcept {
    mask_.clear();
    last_n_ = 0;
    last_c_ = 0;
  }

 private:
  T drop_prob_;
  T keep_prob_;
  T scale_;

  std::mt19937_64 rng_;
  std::bernoulli_distribution bernoulli_;

  // one mask per (n, c)
  std::vector<std::uint8_t> mask_;
  SizeType last_n_ = 0;
  SizeType last_c_ = 0;
};

}  // namespace tinynn
