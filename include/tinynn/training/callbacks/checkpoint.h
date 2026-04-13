#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include <tinynn/training/callback.h>
#include <tinynn/training/callbacks/checkpoint_state_io.h>
#include <tinynn/training/callbacks/checkpoint_stream.h>

namespace tinynn {

enum class CheckpointMode {
  kEveryEpoch,
  kBestEvalLoss,
  kBestEvalAccuracy,
};

// Saves full training checkpoints via save_full_checkpoint().
template <class T>
class CheckpointCallback final : public TrainerCallback<T> {
 public:
  explicit CheckpointCallback(std::string path) : path_(std::move(path)) {}

  void on_fit_begin(Trainer<T>&, const FitOptions&) override {
    if (loaded_) return;
    best_metric_ = std::numeric_limits<T>::infinity();
    has_best_ = false;
    best_path_.clear();
  }

  void on_epoch_end(Trainer<T>& trainer, int epoch, const EpochResult<T>& train,
                    const EpochResult<T>& eval) override {
    const T metric = eval.loss;  // monitored metric

    if (metric < best_metric_) {
      best_metric_ = metric;

      save_checkpoint(trainer, epoch + 1);

      has_best_ = true;
      best_path_ = path_;
    }
  }

  const char* checkpoint_tag() const noexcept override {
      return "CheckpointCallback";
  }

  void save_state(CheckpointWriter& w) const override {
    w.write_u8(static_cast<uint8_t>(has_best_ ? 1 : 0));
    w.write_pod<T>(best_metric_, "best_metric");
    w.write_string(best_path_);
  }

  void load_state(CheckpointReader& r) override {
    loaded_ = true;
    has_best_ = (r.read_u8() != 0);
    best_metric_ = r.read_pod<T>("best_metric");
    best_path_ = r.read_string();
  }

  // ---------------------
  // getters
  // ---------------------
  bool has_best() const noexcept { return has_best_; }

  const std::string& best_path() const noexcept { return best_path_; }

 private:
  void save_checkpoint(Trainer<T>& trainer, int next_epoch) {
    // Full checkpoint (model params + optimizer + callback states + trainer state)
    save_full_checkpoint<T>(path_, trainer, next_epoch);
  }

  std::string path_;
  bool loaded_ = false;
  T best_metric_ = std::numeric_limits<T>::infinity();

  bool has_best_ = false;
  std::string best_path_;
};

}  // namespace tinynn
