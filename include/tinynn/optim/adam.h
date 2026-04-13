#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <tinynn/optim/optimizer.h>
#include <tinynn/optim/weight_decay_filter.h>
#include <tinynn/training/callbacks/checkpoint_io_utils.h>
#include <tinynn/training/callbacks/checkpoint_stream.h>

namespace tinynn {

// Adam optimizer (L2-style weight_decay like SGD; AdamW is separate).
template <class T>
class Adam final : public Optimizer<T> {
 public:
  explicit Adam(T lr = static_cast<T>(1e-3),
                T beta1 = static_cast<T>(0.9),
                T beta2 = static_cast<T>(0.999),
                T eps = static_cast<T>(1e-8),
                T weight_decay = static_cast<T>(0))
      : Optimizer<T>(lr, weight_decay),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps) {
    validate();
  }

  T beta1() const noexcept { return beta1_; }
  T beta2() const noexcept { return beta2_; }
  T eps() const noexcept { return eps_; }

  void set_beta1(T v) { beta1_ = v; validate(); }
  void set_beta2(T v) { beta2_ = v; validate(); }
  void set_eps(T v) { eps_ = v; validate(); }

  void clear_state() {
    m_.clear();
    v_.clear();
    t_ = 0;
  }

  void step(std::span<ParameterView<T>> params) override {
    // Increment time step once per optimizer step (global).
    ++t_;

    const T lr = this->lr_;
    const T wd = this->weight_decay_;
    const T b1 = beta1_;
    const T b2 = beta2_;
    const T eps = eps_;

    // Bias-correction factors
    const T b1t = std::pow(b1, static_cast<T>(t_));
    const T b2t = std::pow(b2, static_cast<T>(t_));
    const T inv_one_minus_b1t = static_cast<T>(1) / (static_cast<T>(1) - b1t);
    const T inv_one_minus_b2t = static_cast<T>(1) / (static_cast<T>(1) - b2t);

    for (auto& pv : params) {
      auto p = pv.param;
      auto g = pv.grad;

      if (p.size() != g.size()) {
        throw std::invalid_argument(
            "tinynn::Adam::step: param/grad size mismatch");
      }
      if (pv.id == 0) {
        throw std::invalid_argument(
            "tinynn::Adam::step: ParameterView.id must be non-zero");
      }

      auto& m = m_[pv.id];
      auto& v = v_[pv.id];
      if (m.size() != static_cast<size_t>(p.size())) {
        m.assign(static_cast<size_t>(p.size()), static_cast<T>(0));
      }
      if (v.size() != static_cast<size_t>(p.size())) {
        v.assign(static_cast<size_t>(p.size()), static_cast<T>(0));
      }

      const bool apply_wd =
        (wd != static_cast<T>(0)) && should_apply_weight_decay(pv);

      for (SizeType i = 0; i < p.size(); ++i) {
        T grad = g[i];
        if (apply_wd) {
          // L2 regularization style (same as SGD/Momentum policy).
          grad += wd * p[i];
        }

        const size_t idx = static_cast<size_t>(i);

        // m, v update
        m[idx] = b1 * m[idx] + (static_cast<T>(1) - b1) * grad;
        v[idx] = b2 * v[idx] + (static_cast<T>(1) - b2) * (grad * grad);

        // bias-corrected
        const T m_hat = m[idx] * inv_one_minus_b1t;
        const T v_hat = v[idx] * inv_one_minus_b2t;

        p[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
      }
    }
  }

  const char* checkpoint_tag() const noexcept override { return "Adam"; }

  void save_state(CheckpointWriter& w) const override {
    w.write_u32(1);  // version
    tinynn::ckpt_write_scalar<T>(w, this->lr_, "lr");
    tinynn::ckpt_write_scalar<T>(w, this->weight_decay_, "wd");
    tinynn::ckpt_write_scalar<T>(w, beta1_, "beta1");
    tinynn::ckpt_write_scalar<T>(w, beta2_, "beta2");
    tinynn::ckpt_write_scalar<T>(w, eps_, "eps");
    w.write_u64(t_);
    tinynn::ckpt_write_state_map<T>(w, m_, "m");
    tinynn::ckpt_write_state_map<T>(w, v_, "v");
  }

  void load_state(CheckpointReader& r) override {
    const uint32_t ver = r.read_u32();
    if (ver != 1) {
      throw std::invalid_argument(
          "tinynn::Adam::load_state: unsupported version");
    }
    this->lr_ = tinynn::ckpt_read_scalar<T>(r, "lr");
    this->weight_decay_ = tinynn::ckpt_read_scalar<T>(r, "wd");
    beta1_ = tinynn::ckpt_read_scalar<T>(r, "beta1");
    beta2_ = tinynn::ckpt_read_scalar<T>(r, "beta2");
    eps_ = tinynn::ckpt_read_scalar<T>(r, "eps");
    t_ = r.read_u64();
    tinynn::ckpt_read_state_map<T>(r, m_, "m");
    tinynn::ckpt_read_state_map<T>(r, v_, "v");
  }

 private:
  void validate() const {
    if (!(beta1_ >= static_cast<T>(0) && beta1_ < static_cast<T>(1))) {
      throw std::invalid_argument("tinynn::Adam: beta1 must be in [0, 1)");
    }
    if (!(beta2_ >= static_cast<T>(0) && beta2_ < static_cast<T>(1))) {
      throw std::invalid_argument("tinynn::Adam: beta2 must be in [0, 1)");
    }
    if (!(eps_ > static_cast<T>(0))) {
      throw std::invalid_argument("tinynn::Adam: eps must be > 0");
    }
  }

  T beta1_ = static_cast<T>(0.9);
  T beta2_ = static_cast<T>(0.999);
  T eps_ = static_cast<T>(1e-8);

  std::uint64_t t_ = 0;  // global step counter

  std::unordered_map<std::uintptr_t, std::vector<T>> m_;
  std::unordered_map<std::uintptr_t, std::vector<T>> v_;
};

}  // namespace tinynn
