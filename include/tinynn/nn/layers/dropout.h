#pragma once

#include <cassert>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include <tinynn/nn/layer.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

template <class T>
class Dropout final : public Layer<T> {
 public:
  explicit Dropout(T drop_prob, std::uint64_t seed = 0)
      : p_(drop_prob),
        base_seed_(seed ? seed : make_default_seed_()),
        rng_(base_seed_),
        keep_(static_cast<double>(1.0 - static_cast<double>(drop_prob))) {
    validate_();
    update_scale_();
  }

  Shape output_shape(const Shape& input_shape) const override {
    return input_shape;
  }

  // ---- reproducibility controls ----
  void set_seed(std::uint64_t seed) {
    base_seed_ = seed ? seed : make_default_seed_();
    rng_.seed(base_seed_);
  }

  // Optional: call once per epoch from Trainer (if you want epoch-dependent dropout)
  void reseed_for_epoch(std::uint64_t global_seed, std::uint64_t epoch) {
    const std::uint64_t s = mix_seed_(global_seed, mix_seed_(base_seed_, epoch));
    rng_.seed(s);
  }

  void forward(ConstTensorView<T> x, TensorView<T> y) override {
    if (x.size() != y.size()) {
      throw std::invalid_argument("tinynn::Dropout::forward: x/y size mismatch");
    }

    // eval or p==0: passthrough
    if (this->phase() == Phase::Eval || p_ == static_cast<T>(0)) {
      copy_(x, y);
      return;
    }

    // train: y = x * mask * scale
    ensure_mask_(x.size());

    const T* xp = x.data();
    T* yp = y.data();

    for (std::size_t i = 0; i < x.size(); ++i) {
      const bool keep = keep_(rng_);
      mask_[i] = static_cast<std::uint8_t>(keep ? 1 : 0);
      yp[i] = keep ? (xp[i] * scale_) : static_cast<T>(0);
    }
  }

  void backward(ConstTensorView<T> dy, TensorView<T> dx) override {
    if (dy.size() != dx.size()) {
      throw std::invalid_argument("tinynn::Dropout::backward: dy/dx size mismatch");
    }

    // eval or p==0: passthrough
    if (this->phase() == Phase::Eval || p_ == static_cast<T>(0)) {
      copy_(dy, dx);
      return;
    }

    if (mask_.size() != dy.size()) {
      assert(false &&
             "Dropout::backward called without matching forward (mask size mismatch)");
      throw std::logic_error("tinynn::Dropout::backward: mask not initialized (forward missing?)");
    }

    const T* dyp = dy.data();
    T* dxp = dx.data();

    for (std::size_t i = 0; i < dy.size(); ++i) {
      dxp[i] = mask_[i] ? (dyp[i] * scale_) : static_cast<T>(0);
    }

    // Clear mask so that "backward without forward" is detectable
    // even when batch size happens to repeat.
    // Capacity is retained by std::vector.
    clear_mask_();
  }

  void set_drop_prob(T drop_prob) {
    p_ = drop_prob;
    validate_();
    update_scale_();
    keep_ = std::bernoulli_distribution(
        static_cast<double>(1.0 - static_cast<double>(p_)));
  }

  [[nodiscard]] T drop_prob() const noexcept { return p_; }

 private:
  T p_{0};
  T scale_{1};

  std::uint64_t base_seed_{0};
  std::mt19937_64 rng_;
  std::bernoulli_distribution keep_;
  std::vector<std::uint8_t> mask_;

  void validate_() const {
    if (!(p_ >= static_cast<T>(0) && p_ < static_cast<T>(1))) {
      throw std::invalid_argument("tinynn::Dropout: drop_prob must be in [0, 1)");
    }
  }

  void update_scale_() {
    scale_ = (p_ == static_cast<T>(0))
                 ? static_cast<T>(1)
                 : (static_cast<T>(1) / (static_cast<T>(1) - p_));
  }

  void ensure_mask_(std::size_t n) {
    if (mask_.size() != n) mask_.assign(n, 0);
  }

  void clear_mask_() noexcept {
    // Clear size (retain capacity) so forward-missing is detectable.
    mask_.clear();
  }

  static std::uint64_t mix_seed_(std::uint64_t a, std::uint64_t b) {
    // splitmix64-style mix
    std::uint64_t x = a ^ (b + 0x9e3779b97f4a7c15ULL);
    x ^= (x >> 30);
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= (x >> 27);
    x *= 0x94d049bb133111ebULL;
    x ^= (x >> 31);
    return x;
  }

  static std::uint64_t make_default_seed_() {
    std::random_device rd;
    return (static_cast<std::uint64_t>(rd()) << 32) ^
           static_cast<std::uint64_t>(rd());
  }

  static void copy_(ConstTensorView<T> src, TensorView<T> dst) {
    const T* sp = src.data();
    T* dp = dst.data();
    for (std::size_t i = 0; i < src.size(); ++i) dp[i] = sp[i];
  }
};

}  // namespace tinynn
