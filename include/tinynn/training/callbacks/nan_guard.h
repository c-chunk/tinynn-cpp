#pragma once

#include <iostream>

#include <tinynn/training/callback.h>
#include <tinynn/training/trainer_types.h>

namespace tinynn {

template <class T>
class NaNGuardCallback final : public TrainerCallback<T> {
 public:
  explicit NaNGuardCallback(bool verbose = true)
      : verbose_(verbose) {}

  void on_train_step_end(
      Trainer<T>& trainer,
      int epoch,
      SizeType step,
      const StepResult<T>& res) override {

    if (invalid_(res.loss) || invalid_(res.accuracy)) {
      report_("train", epoch, step, res.loss, res.accuracy);
      trainer.request_stop();
    }
  }

  void on_eval_step_end(
      Trainer<T>& trainer,
      int epoch,
      SizeType step,
      const StepResult<T>& res) override {

    if (invalid_(res.loss) || invalid_(res.accuracy)) {
      report_("eval", epoch, step, res.loss, res.accuracy);
      trainer.request_stop();
    }
  }

 private:
  bool verbose_;

  static bool invalid_(T v) {
    return std::isnan(v) || std::isinf(v);
  }

  void report_(
      const char* phase,
      int epoch,
      SizeType step,
      T loss,
      T acc) const {

    if (!verbose_) return;

    std::cerr
      << "[NaNGuard] "
      << phase
      << " invalid metric detected "
      << "epoch=" << epoch
      << " step=" << step
      << " loss=" << loss
      << " acc=" << acc
      << "\n";
  }
};

}  // namespace tinynn
