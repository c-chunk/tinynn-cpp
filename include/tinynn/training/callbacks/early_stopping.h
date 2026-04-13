#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>  // std::move

#include <tinynn/training/callback.h>
#include <tinynn/training/trainer_types.h>

// Required when restore_best is enabled.
#include <tinynn/training/callbacks/checkpoint_io.h>

namespace tinynn {

enum class EarlyStopMonitor {
  EvalLoss,
  EvalAccuracy,
};

template <class T>
class EarlyStoppingCallback final : public TrainerCallback<T> {
 public:
  struct Options {
    EarlyStopMonitor monitor = EarlyStopMonitor::EvalLoss;
    SizeType patience = 5;            // Stop after this many bad epochs.
    T min_delta = static_cast<T>(0);  // Minimum improvement threshold.
    bool restore_best = false;        // Restore best parameters on stop.

    // Path used when restore_best is enabled.
    std::string best_path = "best.bin";
  };

  explicit EarlyStoppingCallback(Options opt) : opt_(std::move(opt)) {
    if (opt_.patience == 0) {
      throw std::invalid_argument(
          "EarlyStoppingCallback: patience must be > 0");
    }
    if (opt_.restore_best && opt_.best_path.empty()) {
      throw std::invalid_argument(
          "EarlyStoppingCallback: best_path is empty");
    }
    reset_state_();
  }

  void on_fit_begin(Trainer<T>& /*trainer*/,
                    const FitOptions& /*opt*/) override {
    reset_state_();
  }

  void on_epoch_end(Trainer<T>& trainer, int /*epoch*/,
                    const EpochResult<T>& /*tr*/,
                    const EpochResult<T>& ev) override {
    const T current = monitored_value_(ev);

    // Treat NaN as "no improvement". 
    // Stopping immediately would also be a valid policy.
    if (!(current == current)) {  // NaN check
      ++bad_epochs_;
      if (bad_epochs_ >= opt_.patience) {
        if (opt_.restore_best) {
          load_checkpoint_params<T>(opt_.best_path, trainer.model());
        }
        trainer.request_stop();
      }
      return;
    }

    if (!has_best_) {
      has_best_ = true;
      best_ = current;
      bad_epochs_ = 0;
      return;
    }

    if (is_improved_(current, best_)) {
      best_ = current;
      bad_epochs_ = 0;
      return;
    }

    ++bad_epochs_;
    if (bad_epochs_ >= opt_.patience) {
      if (opt_.restore_best) {
        load_checkpoint_params<T>(opt_.best_path, trainer.model());
      }
      trainer.request_stop();
    }
  }

  [[nodiscard]] bool has_best() const noexcept { return has_best_; }
  [[nodiscard]] T best_value() const noexcept { return best_; }
  [[nodiscard]] SizeType bad_epochs() const noexcept { return bad_epochs_; }

 private:
  Options opt_{};

  bool has_best_ = false;
  T best_ = static_cast<T>(0);
  SizeType bad_epochs_ = 0;

  void reset_state_() noexcept {
    has_best_ = false;
    bad_epochs_ = 0;
    if (opt_.monitor == EarlyStopMonitor::EvalLoss) {
      best_ = std::numeric_limits<T>::infinity();
    } else {
      best_ = -std::numeric_limits<T>::infinity();
    }
  }

  T monitored_value_(const EpochResult<T>& ev) const {
    return (opt_.monitor == EarlyStopMonitor::EvalLoss) ? ev.loss : ev.accuracy;
  }

  bool is_improved_(T current, T best) const {
    if (opt_.monitor == EarlyStopMonitor::EvalLoss) {
      return current < (best - opt_.min_delta);
    }
    return current > (best + opt_.min_delta);
  }
};

}  // namespace tinynn
