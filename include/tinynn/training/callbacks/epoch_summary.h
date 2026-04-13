#pragma once

#include <iostream>

#include <tinynn/training/callback.h>
#include <tinynn/training/trainer_types.h>

namespace tinynn {

template <class T>
class EpochSummaryCallback final : public TrainerCallback<T> {
 public:
  explicit EpochSummaryCallback(bool enabled = true)
      : enabled_(enabled) {}

  void on_epoch_end(
      Trainer<T>&,
      int epoch,
      const EpochResult<T>& tr,
      const EpochResult<T>& ev) override {

    if (!enabled_) return;

    std::cout
      << "epoch " << epoch
      << " train loss=" << tr.loss
      << " acc=" << tr.accuracy
      << " (" << tr.num_correct << "/" << tr.num_samples << ")"
      << " | eval loss=" << ev.loss
      << " acc=" << ev.accuracy
      << " (" << ev.num_correct << "/" << ev.num_samples << ")"
      << "\n" << std::flush;
  }

 private:
  bool enabled_ = true;
};

}  // namespace tinynn
