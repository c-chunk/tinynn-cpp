#pragma once

#include <span>
#include <stdexcept>

#include <tinynn/optim/optimizer.h>
#include <tinynn/optim/weight_decay_filter.h>
#include <tinynn/training/callbacks/checkpoint_io_utils.h>
#include <tinynn/training/callbacks/checkpoint_stream.h>

namespace tinynn {

// Plain SGD (no momentum).
template <class T>
class SGD final : public Optimizer<T> {
 public:
  explicit SGD(T lr = static_cast<T>(1e-2), T weight_decay = static_cast<T>(0))
      : Optimizer<T>(lr, weight_decay) {}

  void step(std::span<ParameterView<T>> params) override {
    const T lr = this->lr_;
    const T wd = this->weight_decay_;

    for (auto& pv : params) {
      auto p = pv.param;  // TensorView<T>
      auto g = pv.grad;   // TensorView<const T>

      if (p.size() != g.size()) {
        throw std::invalid_argument("tinynn::SGD::step: param/grad size mismatch");
      }

      const bool apply_wd =
          (wd != static_cast<T>(0)) && should_apply_weight_decay(pv);

      for (SizeType i = 0; i < p.size(); ++i) {
        T grad = g[i];
        if (apply_wd) {
          grad += wd * p[i];
        }
        p[i] -= lr * grad;
      }
    }
  }

  const char* checkpoint_tag() const noexcept override { return "SGD"; }

  void save_state(CheckpointWriter& w) const override {
    w.write_u32(1);  // version
    tinynn::ckpt_write_scalar<T>(w, this->lr_, "lr");
    tinynn::ckpt_write_scalar<T>(w, this->weight_decay_, "wd");
  }

  void load_state(CheckpointReader& r) override {
    const uint32_t ver = r.read_u32();
    if (ver != 1) {
      throw std::invalid_argument(
          "tinynn::SGD::load_state: unsupported version");
    }
    this->lr_ = tinynn::ckpt_read_scalar<T>(r, "lr");
    this->weight_decay_ = tinynn::ckpt_read_scalar<T>(r, "wd");
  }
};

}  // namespace tinynn
