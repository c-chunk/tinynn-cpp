#pragma once

#include <limits>
#include <stdexcept>

#include <tinynn/training/callback.h>
#include <tinynn/training/trainer_types.h>

namespace tinynn {

enum class PlateauMonitor {
  EvalLoss,
  EvalAccuracy,
};

template <class T>
class ReduceLROnPlateauCallback final : public TrainerCallback<T> {
 public:
  struct Options {
    PlateauMonitor monitor = PlateauMonitor::EvalLoss;

    T factor = static_cast<T>(0.1);   // lr *= factor
    SizeType patience = 5;            // Reduce lr after this many bad epochs.
    T min_delta = static_cast<T>(0);  // Minimum improvement threshold.
    SizeType cooldown = 0;            // Wait this many epochs after an lr update.
    T min_lr = static_cast<T>(0);     // Lower bound for learning rate.
  };

  explicit ReduceLROnPlateauCallback(Options opt) : opt_(opt) {
    if (!(opt_.factor > static_cast<T>(0)) ||
        !(opt_.factor < static_cast<T>(1))) {
      throw std::invalid_argument("ReduceLROnPlateau: factor must be in (0,1)");
    }
    if (opt_.patience == 0) {
      throw std::invalid_argument("ReduceLROnPlateau: patience must be > 0");
    }
    if (opt_.min_lr < static_cast<T>(0)) {
      throw std::invalid_argument("ReduceLROnPlateau: min_lr must be >= 0");
    }
    reset_();
  }

  void on_fit_begin(Trainer<T>& /*trainer*/, const FitOptions& /*opt*/) override {
    reset_();
  }

  void on_epoch_end(Trainer<T>& trainer, int /*epoch*/,
                    const EpochResult<T>& /*tr*/,
                    const EpochResult<T>& ev) override {
    const T current = monitored_value_(ev);

    if (!has_best_) {
      has_best_ = true;
      best_ = current;
      bad_epochs_ = 0;
      return;
    }

    // During cooldown, do not advance bad_epochs.
    if (cooldown_left_ > 0) {
      --cooldown_left_;
      // Still allow best_ to improve during cooldown.
      if (is_improved_(current, best_)) {
        best_ = current;
        bad_epochs_ = 0;
      }
      return;
    }

    if (is_improved_(current, best_)) {
      best_ = current;
      bad_epochs_ = 0;
      return;
    }

    ++bad_epochs_;
    if (bad_epochs_ >= opt_.patience) {
      auto& opt = trainer.optimizer();
      const T cur_lr = opt.learning_rate();
      T next_lr = cur_lr * opt_.factor;
      if (next_lr < opt_.min_lr) next_lr = opt_.min_lr;

      // Do nothing if lr would not change.
      if (next_lr < cur_lr) {
        opt.set_learning_rate(next_lr);
        cooldown_left_ = opt_.cooldown;
      }

      bad_epochs_ = 0;  // Reset, following the common scheduler behavior.
    }
  }

  [[nodiscard]] bool has_best() const noexcept { return has_best_; }
  [[nodiscard]] T best_value() const noexcept { return best_; }
  [[nodiscard]] SizeType bad_epochs() const noexcept { return bad_epochs_; }
  [[nodiscard]] SizeType cooldown_left() const noexcept { return cooldown_left_; }

 private:
  Options opt_{};

  bool has_best_ = false;
  T best_ = static_cast<T>(0);
  SizeType bad_epochs_ = 0;
  SizeType cooldown_left_ = 0;

  void reset_() noexcept {
    has_best_ = false;
    bad_epochs_ = 0;
    cooldown_left_ = 0;
    if (opt_.monitor == PlateauMonitor::EvalLoss) {
      best_ = std::numeric_limits<T>::infinity();
    } else {
      best_ = -std::numeric_limits<T>::infinity();
    }
  }

  T monitored_value_(const EpochResult<T>& ev) const {
    return (opt_.monitor == PlateauMonitor::EvalLoss) ? ev.loss : ev.accuracy;
  }

  bool is_improved_(T current, T best) const {
    if (opt_.monitor == PlateauMonitor::EvalLoss) {
      // Loss: smaller is better
      return current < (best - opt_.min_delta);
    }
    // Accuracy: larger is better
    return current > (best + opt_.min_delta);
  }
};

}  // namespace tinynn
