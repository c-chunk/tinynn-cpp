#pragma once

#include <stdexcept>

#include <tinynn/training/callback.h>
#include <tinynn/training/trainer_types.h>

namespace tinynn {

template <class T>
class StepLRCallback final : public TrainerCallback<T> {
 public:
  struct Options {
    int step_size = 1;              // Update every step_size epochs (>= 1).
    T gamma = static_cast<T>(0.1);  // lr *= gamma
    T min_lr = static_cast<T>(0);   // Lower bound for learning rate.
  };

  explicit StepLRCallback(Options opt) : opt_(opt) {
    if (opt_.step_size <= 0) {
      throw std::invalid_argument("StepLRCallback: step_size must be >= 1");
    }
    if (!(opt_.gamma > static_cast<T>(0))) {
      throw std::invalid_argument("StepLRCallback: gamma must be > 0");
    }
    if (opt_.min_lr < static_cast<T>(0)) {
      throw std::invalid_argument("StepLRCallback: min_lr must be >= 0");
    }
  }

  void on_epoch_end(Trainer<T>& trainer, int epoch,
                    const EpochResult<T>& /*tr*/,
                    const EpochResult<T>& /*ev*/) override {
    // Epoch is 0-based. Since the update happens after an epoch finishes,
    // use (epoch + 1) here.
    const int e = epoch + 1;
    if (e % opt_.step_size != 0) return;

    auto& opt = trainer.optimizer();

    const T cur = opt.learning_rate();
    T next = cur * opt_.gamma;
    if (next < opt_.min_lr) next = opt_.min_lr;

    if (next == cur) return;
    opt.set_learning_rate(next);
  }

 private:
  Options opt_{};
};

}  // namespace tinynn
