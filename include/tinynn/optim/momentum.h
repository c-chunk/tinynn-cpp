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

// SGD with Momentum (no Nesterov).
template <class T>
class Momentum final : public Optimizer<T> {
 public:
  explicit Momentum(T lr = static_cast<T>(1e-2),
                    T momentum = static_cast<T>(0.9),
                    T weight_decay = static_cast<T>(0))
      : Optimizer<T>(lr, weight_decay), momentum_(momentum) {
    if (!(momentum_ >= static_cast<T>(0)) || !(momentum_ < static_cast<T>(1))) {
      throw std::invalid_argument(
          "tinynn::Momentum: momentum must be in [0, 1)");
    }
  }

  T momentum() const noexcept { return momentum_; }
  void set_momentum(T m) {
    if (!(m >= static_cast<T>(0)) || !(m < static_cast<T>(1))) {
      throw std::invalid_argument(
          "tinynn::Momentum: momentum must be in [0, 1)");
    }
    momentum_ = m;
  }

  void clear_state() { velocity_.clear(); }

  void step(std::span<ParameterView<T>> params) override {
    const T lr = this->lr_;
    const T wd = this->weight_decay_;
    const T mu = momentum_;

    for (auto& pv : params) {
      auto p = pv.param;  // TensorView<T>
      auto g = pv.grad;   // TensorView<const T>

      if (p.size() != g.size()) {
        throw std::invalid_argument("tinynn::Momentum::step: param/grad size mismatch");
      }
      if (pv.id == 0) {
        throw std::invalid_argument("tinynn::Momentum::step: ParameterView.id must be non-zero");
      }

      auto& v = velocity_[pv.id];
      if (v.size() != static_cast<size_t>(p.size())) {
        v.assign(static_cast<size_t>(p.size()), static_cast<T>(0));
      }

      const bool apply_wd =
          (wd != static_cast<T>(0)) && should_apply_weight_decay(pv);

      for (SizeType i = 0; i < p.size(); ++i) {
        T grad = g[i];
        if (apply_wd) {
          grad += wd * p[i];  // L2 regularization (same policy as SGD)
        }

        const size_t idx = static_cast<size_t>(i);
        v[idx] = mu * v[idx] + grad;  // velocity update
        p[i] -= lr * v[idx];          // parameter update
      }
    }
  }

  const char* checkpoint_tag() const noexcept override { return "Momentum"; }

  void save_state(CheckpointWriter& w) const override {
    w.write_u32(1);  // version
    tinynn::ckpt_write_scalar<T>(w, this->lr_, "lr");
    tinynn::ckpt_write_scalar<T>(w, this->weight_decay_, "wd");
    tinynn::ckpt_write_scalar<T>(w, momentum_, "momentum");
    tinynn::ckpt_write_state_map<T>(w, velocity_, "velocity");
  }

  void load_state(CheckpointReader& r) override {
    const uint32_t ver = r.read_u32();
    if (ver != 1)
      throw std::invalid_argument(
          "tinynn::Momentum::load_state: unsupported version");
    this->lr_ = tinynn::ckpt_read_scalar<T>(r, "lr");
    this->weight_decay_ = tinynn::ckpt_read_scalar<T>(r, "wd");
    momentum_ = tinynn::ckpt_read_scalar<T>(r, "momentum");
    tinynn::ckpt_read_state_map<T>(r, velocity_, "velocity");
  }

 private:
  T momentum_ = static_cast<T>(0.9);

  // key: ParameterView.id
  std::unordered_map<std::uintptr_t, std::vector<T>> velocity_;
};

}  // namespace tinynn
