#pragma once

#include <span>
#include <stdexcept>

#include <tinynn/nn/parameter_view.h>

namespace tinynn {

class CheckpointWriter;
class CheckpointReader;

template <class T>
class Optimizer {
 public:
  virtual ~Optimizer() noexcept = default;

  // Hyper-params (common)
  T learning_rate() const noexcept { return lr_; }
  void set_learning_rate(T lr) {
    if (!(lr > static_cast<T>(0))) {
      throw std::invalid_argument("tinynn::Optimizer: learning_rate must be > 0");
    }
    lr_ = lr;
  }

  // L2 regularization (weight decay)
  T weight_decay() const noexcept { return weight_decay_; }
  void set_weight_decay(T wd) {
    if (wd < static_cast<T>(0)) {
      throw std::invalid_argument("tinynn::Optimizer: weight_decay must be >= 0");
    }
    weight_decay_ = wd;
  }

  // Checkpointing (optional but recommended for exact resume).
  // Returns a stable type tag (e.g., "SGD", "AdamW").
  virtual const char* checkpoint_tag() const noexcept = 0;

  // Save/load optimizer internal state (e.g., momentum buffers).
  virtual void save_state(CheckpointWriter& /*w*/) const {}
  virtual void load_state(CheckpointReader& /*r*/) {}

  // Perform one optimization step.
  // params: list of (param, grad, kind).
  virtual void step(std::span<ParameterView<T>> params) = 0;

 protected:
  explicit Optimizer(T lr = static_cast<T>(1e-2), T weight_decay = static_cast<T>(0))
      : lr_(lr), weight_decay_(weight_decay) {
    if (!(lr_ > static_cast<T>(0))) {
      throw std::invalid_argument("tinynn::Optimizer: learning_rate must be > 0");
    }
    if (weight_decay_ < static_cast<T>(0)) {
      throw std::invalid_argument("tinynn::Optimizer: weight_decay must be >= 0");
    }
  }

  T lr_ = static_cast<T>(1e-2);
  T weight_decay_ = static_cast<T>(0);
};

}  // namespace tinynn
