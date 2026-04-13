#pragma once

#include <tinynn/tinynn.h>
#include <tinynn/training/trainer_types.h>

#include <exception>  // std::exception_ptr

namespace tinynn {

template <class T>
class Trainer;

// For checkpoint state I/O (forward declarations).
class CheckpointWriter;
class CheckpointReader;

template <class T>
class TrainerCallback {
 public:
  virtual ~TrainerCallback() = default;

  // ----------------
  // fit
  // ----------------
  virtual void on_fit_begin(Trainer<T>& /*trainer*/,
                            const FitOptions& /*opt*/) {}
  virtual void on_fit_end(Trainer<T>& /*trainer*/, const FitOptions& /*opt*/) {}

  // ----------------
  // epoch
  // ----------------
  virtual void on_epoch_begin(Trainer<T>& /*trainer*/, int /*epoch*/) {}
  virtual void on_epoch_end(Trainer<T>& /*trainer*/, int /*epoch*/,
                            const EpochResult<T>& /*train*/,
                            const EpochResult<T>& /*eval*/) {}

  // ----------------
  // train batch/step
  // ----------------
  virtual void on_train_batch_begin(Trainer<T>& /*trainer*/, int /*epoch*/,
                                    SizeType /*step*/) {}

  virtual void on_train_step_end(Trainer<T>& /*trainer*/, int /*epoch*/,
                                 SizeType /*step*/,
                                 const StepResult<T>& /*res*/) {}

  virtual void on_train_batch_end(Trainer<T>& /*trainer*/, int /*epoch*/,
                                  SizeType /*step*/) {}

  // ----------------
  // eval batch/step
  // ----------------
  virtual void on_eval_batch_begin(Trainer<T>& /*trainer*/, int /*epoch*/,
                                   SizeType /*step*/) {}

  virtual void on_eval_step_end(Trainer<T>& /*trainer*/, int /*epoch*/,
                                SizeType /*step*/,
                                const StepResult<T>& /*res*/) {}

  virtual void on_eval_batch_end(Trainer<T>& /*trainer*/, int /*epoch*/,
                                 SizeType /*step*/) {}

  // ----------------
  // checkpoint state (optional)
  // ----------------
  // Unique stable tag for checkpointing. Return nullptr if this callback has no state.
  virtual const char* checkpoint_tag() const noexcept { return nullptr; }

  // Save/load minimal internal state required for exact resume.
  virtual void save_state(CheckpointWriter& /*w*/) const {}
  virtual void load_state(CheckpointReader& /*r*/) {}

  // ----------------
  // error
  // ----------------
  virtual void on_exception(Trainer<T>& /*trainer*/,
                            std::exception_ptr /*ep*/) {}
};

}  // namespace tinynn
