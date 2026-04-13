#pragma once

#include <iostream>
#include <stdexcept>

#include <tinynn/training/callback.h>
#include <tinynn/training/trainer_types.h>

namespace tinynn {

template <class T>
class PrintEveryCallback final : public TrainerCallback<T> {
 public:
  explicit PrintEveryCallback(SizeType every_steps) : every_(every_steps) {
    if (every_ == 0) {
      throw std::invalid_argument(
          "tinynn::PrintEveryCallback: every_steps must be > 0");
    }
  }

  void on_train_step_end(Trainer<T>& /*trainer*/, int epoch, SizeType step,
                         const StepResult<T>& res) override {
    if (step % every_ != 0) return;

    std::cout << "[train] epoch=" << epoch << " step=" << step
              << " loss=" << res.loss << " acc=" << res.accuracy << " ("
              << res.num_correct << "/" << res.batch_size << ")\n";
  }

  void on_eval_step_end(Trainer<T>& /*trainer*/, int epoch, SizeType step,
                        const StepResult<T>& res) override {
    if (step % every_ != 0) return;

    std::cout << "[eval ] epoch=" << epoch << " step=" << step
              << " loss=" << res.loss << " acc=" << res.accuracy << " ("
              << res.num_correct << "/" << res.batch_size << ")\n";
  }

 private:
  SizeType every_;
};

}  // namespace tinynn
