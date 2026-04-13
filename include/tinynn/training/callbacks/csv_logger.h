#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <tinynn/training/callback.h>
#include <tinynn/training/trainer_types.h>

namespace tinynn {

template <class T>
class CSVLoggerCallback final : public TrainerCallback<T> {
 public:
  explicit CSVLoggerCallback(std::string path,
                             bool append = false)
      : path_(std::move(path)), append_(append) {}

  void on_fit_begin(Trainer<T>&, const FitOptions&) override {
    open_file_();
  }

  void on_epoch_end(Trainer<T>&,
                    int epoch,
                    const EpochResult<T>& tr,
                    const EpochResult<T>& ev) override {
    if (!ofs_) return;

    ofs_
        << epoch << ","
        << tr.loss << ","
        << tr.accuracy << ","
        << ev.loss << ","
        << ev.accuracy
        << "\n";

    ofs_.flush();
  }

 private:
  void open_file_() {
    std::ios::openmode mode = std::ios::out;
    if (append_) mode |= std::ios::app;

    ofs_.open(path_, mode);
    if (!ofs_) {
      throw std::runtime_error(
          "CSVLoggerCallback: cannot open file: " + path_);
    }

    if (!append_) {
      // CSV header
      ofs_ << "epoch,train_loss,train_acc,eval_loss,eval_acc\n";
    }
  }

  std::string path_;
  bool append_;
  std::ofstream ofs_;
};

}  // namespace tinynn
