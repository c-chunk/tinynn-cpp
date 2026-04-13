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

// RMSProp optimizer (mean-square only; no momentum).
template <class T>
class RMSProp final : public Optimizer<T> {
 public:
  explicit RMSProp(T lr = static_cast<T>(1e-3),
                   T rho = static_cast<T>(0.99),
                   T eps = static_cast<T>(1e-8),
                   T weight_decay = static_cast<T>(0))
      : Optimizer<T>(lr, weight_decay), rho_(rho), eps_(eps) {
    validate();
  }

  T rho() const noexcept { return rho_; }
  T eps() const noexcept { return eps_; }

  void set_rho(T v) { rho_ = v; validate(); }
  void set_eps(T v) { eps_ = v; validate(); }

  void clear_state() { ms_.clear(); }

  void step(std::span<ParameterView<T>> params) override {
    const T lr = this->lr_;
    const T wd = this->weight_decay_;
    const T rho = rho_;
    const T eps = eps_;

    for (auto& pv : params) {
      auto p = pv.param;
      auto g = pv.grad;

      if (p.size() != g.size()) {
        throw std::invalid_argument(
            "tinynn::RMSProp::step: param/grad size mismatch");
      }
      if (pv.id == 0) {
        throw std::invalid_argument(
            "tinynn::RMSProp::step: ParameterView.id must be non-zero");
      }

      auto& ms = ms_[pv.id];
      if (ms.size() != static_cast<size_t>(p.size())) {
        ms.assign(static_cast<size_t>(p.size()), static_cast<T>(0));
      }

      const bool apply_wd =
          (wd != static_cast<T>(0)) && should_apply_weight_decay(pv);

      for (SizeType i = 0; i < p.size(); ++i) {
        T grad = g[i];
        if (apply_wd) {
          // L2 regularization style (same policy as SGD/Momentum/Adam)
          grad += wd * p[i];
        }

        const size_t idx = static_cast<size_t>(i);

        // ms = rho*ms + (1-rho)*grad^2
        const T gg = grad * grad;
        ms[idx] = rho * ms[idx] + (static_cast<T>(1) - rho) * gg;

        // p -= lr * grad / (sqrt(ms) + eps)
        p[i] -= lr * grad / (std::sqrt(ms[idx]) + eps);
      }
    }
  }

  const char* checkpoint_tag() const noexcept override { return "RMSProp"; }

  void save_state(CheckpointWriter& w) const override {
    w.write_u32(1);  // version
    tinynn::ckpt_write_scalar<T>(w, this->lr_, "lr");
    tinynn::ckpt_write_scalar<T>(w, this->weight_decay_, "wd");
    tinynn::ckpt_write_scalar<T>(w, rho_, "rho");
    tinynn::ckpt_write_scalar<T>(w, eps_, "eps");
    tinynn::ckpt_write_state_map<T>(w, ms_, "ms");
  }

  void load_state(CheckpointReader& r) override {
    const uint32_t ver = r.read_u32();
    if (ver != 1) {
      throw std::invalid_argument(
          "tinynn::RMSProp::load_state: unsupported version");
    }
    this->lr_ = tinynn::ckpt_read_scalar<T>(r, "lr");
    this->weight_decay_ = tinynn::ckpt_read_scalar<T>(r, "wd");
    rho_ = tinynn::ckpt_read_scalar<T>(r, "rho");
    eps_ = tinynn::ckpt_read_scalar<T>(r, "eps");
    tinynn::ckpt_read_state_map<T>(r, ms_, "ms");
  }

 private:
  void validate() const {
    if (!(rho_ >= static_cast<T>(0) && rho_ < static_cast<T>(1))) {
      throw std::invalid_argument("tinynn::RMSProp: rho must be in [0, 1)");
    }
    if (!(eps_ > static_cast<T>(0))) {
      throw std::invalid_argument("tinynn::RMSProp: eps must be > 0");
    }
  }

  T rho_ = static_cast<T>(0.99);
  T eps_ = static_cast<T>(1e-8);

  std::unordered_map<std::uintptr_t, std::vector<T>> ms_;
};

}  // namespace tinynn
