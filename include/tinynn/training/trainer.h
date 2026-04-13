#pragma once

#include <span>
#include <stdexcept>
#include <vector>

#include <tinynn/nn/sequential.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/nn/predict.h>
#include <tinynn/optim/optimizer.h>
#include <tinynn/training/trainer_types.h>
#include <tinynn/training/callback.h>

namespace tinynn {

namespace detail {

template <class Loader>
auto maybe_set_epoch(Loader& loader, int epoch, int)
    -> decltype(loader.set_epoch(epoch), void()) {
  loader.set_epoch(epoch);
}

template <class Loader>
void maybe_set_epoch(Loader&, int, ...) {}

template <class Loader>
auto maybe_prepare_epoch(Loader& loader, int, int)
    -> decltype(loader.prepare_epoch(), void()) {
  loader.prepare_epoch();
}

template <class Loader>
void maybe_prepare_epoch(Loader&, int, ...) {}

}  // namespace detail

template <class T>
class Trainer {
 public:
  Trainer(Sequential<T>& model,
          SoftmaxWithLoss<T>& loss_fn,
          Optimizer<T>& optimizer)
      : model_(model), loss_fn_(loss_fn), opt_(optimizer) {}

  // One training step (one batch): forward -> loss -> backward -> optimizer.step
  [[nodiscard]] StepResult<T> train_step(ConstTensorView<T> x,
                                        std::span<const SizeType> labels) {
    // 1) forward
    auto logits = model_.forward(x);

    // 2) loss (mean)
    const T loss = loss_fn_.forward(logits, labels);

    // 3) dlogits
    auto dlogits = loss_fn_.backward();  // (p - t) / batch

    // 4) backward
    (void)model_.backward(dlogits);

    // 5) collect params + step
    params_cache_.clear();
    model_.collect_parameter_views(params_cache_);
    opt_.step(std::span<ParameterView<T>>(params_cache_));

    // 6) metrics
    const auto [num_correct, batch] = compute_accuracy_from_logits_(logits, labels);

    StepResult<T> res;
    res.loss = loss;
    res.batch_size = batch;
    res.num_correct = num_correct;
    res.accuracy = (batch == 0) ? static_cast<T>(0)
                                : static_cast<T>(num_correct) / static_cast<T>(batch);
    return res;
  }

  // One eval step (one batch): forward -> loss -> metrics (no backward/step)
  [[nodiscard]] StepResult<T> eval_step(ConstTensorView<T> x,
                                       std::span<const SizeType> labels) {
    auto logits = model_.forward(x);
    const T loss = loss_fn_.forward(logits, labels);

    const auto [num_correct, batch] = compute_accuracy_from_logits_(logits, labels);

    StepResult<T> res;
    res.loss = loss;
    res.batch_size = batch;
    res.num_correct = num_correct;
    res.accuracy = (batch == 0) ? static_cast<T>(0)
                                : static_cast<T>(num_correct) / static_cast<T>(batch);
    return res;
  }

  template <class TrainLoader, class EvalLoader>
  void fit(TrainLoader& train_loader, EvalLoader& eval_loader,
           const FitOptions& opt = {}) {
    if (opt.epochs <= 0) {
      throw std::invalid_argument("tinynn::Trainer::fit: epochs must be > 0");
    }
    if (opt.start_epoch < 0) {
      throw std::invalid_argument("tinynn::Trainer::fit: start_epoch must be >= 0");
    }
    if (opt.start_epoch > opt.epochs) {
      throw std::invalid_argument("tinynn::Trainer::fit: start_epoch must be <= epochs");
    }

    stop_requested_ = false;  // reset stop request at the beginning of fit

    for_each_cb_([&](auto& cb) { cb.on_fit_begin(*this, opt); });
    try {
      for (int epoch = opt.start_epoch; epoch < opt.epochs; ++epoch) {
        for_each_cb_([&](auto& cb) { cb.on_epoch_begin(*this, epoch); });

        const auto tr = train_epoch(train_loader, epoch);
        const auto ev = eval_epoch(eval_loader, epoch);

        for_each_cb_([&](auto& cb) { cb.on_epoch_end(*this, epoch, tr, ev); });

        if (stop_requested_) break;
      }
    } catch (...) {
      for_each_cb_([&](auto& cb) { cb.on_fit_end(*this, opt); });
      throw;
    }
    for_each_cb_([&](auto& cb) { cb.on_fit_end(*this, opt); });
  }

  template <class Loader>
  [[nodiscard]] EpochResult<T> train_epoch(Loader& loader, int epoch) {

    model_.train();  // ensure train mode

    EpochResult<T> out;
    T loss_sum = static_cast<T>(0);

    detail::maybe_set_epoch(loader, epoch, 0);
    detail::maybe_prepare_epoch(loader, epoch, 0);

    SizeType step = 0;
    for (const auto& batch : loader) {
      const SizeType b = static_cast<SizeType>(batch.y.size());

      for_each_cb_(
          [&](auto& cb) { cb.on_train_batch_begin(*this, epoch, step); });

      auto res = train_step(batch.x.view(), std::span<const SizeType>(batch.y));

      for_each_cb_(
          [&](auto& cb) { cb.on_train_step_end(*this, epoch, step, res); });
      for_each_cb_(
          [&](auto& cb) { cb.on_train_batch_end(*this, epoch, step); });

      loss_sum += res.loss * static_cast<T>(b);
      out.num_samples += b;
      out.num_correct += res.num_correct;
      ++step;
      ++global_step_;
    }

    if (out.num_samples > 0) {
      out.loss = loss_sum / static_cast<T>(out.num_samples);
      out.accuracy =
          static_cast<T>(out.num_correct) / static_cast<T>(out.num_samples);
    }
    return out;
  }

  template <class Loader>
  [[nodiscard]] EpochResult<T> eval_epoch(Loader& loader, int epoch) {

    model_.eval();  // ensure eval mode

    EpochResult<T> out;
    T loss_sum = static_cast<T>(0);

    detail::maybe_set_epoch(loader, epoch, 0);
    detail::maybe_prepare_epoch(loader, epoch, 0);

    SizeType step = 0;
    for (const auto& batch : loader) {
      const SizeType b = static_cast<SizeType>(batch.y.size());

      for_each_cb_(
          [&](auto& cb) { cb.on_eval_batch_begin(*this, epoch, step); });

      auto res = eval_step(batch.x.view(), std::span<const SizeType>(batch.y));

      for_each_cb_(
          [&](auto& cb) { cb.on_eval_step_end(*this, epoch, step, res); });
      for_each_cb_([&](auto& cb) { cb.on_eval_batch_end(*this, epoch, step); });

      loss_sum += res.loss * static_cast<T>(b);
      out.num_samples += b;
      out.num_correct += res.num_correct;
      ++step;
    }

    if (out.num_samples > 0) {
      out.loss = loss_sum / static_cast<T>(out.num_samples);
      out.accuracy =
          static_cast<T>(out.num_correct) / static_cast<T>(out.num_samples);
    }
    return out;
  }

  void add_callback(TrainerCallback<T>* cb) {
    if (!cb)
      throw std::invalid_argument("tinynn::Trainer::add_callback: cb is null");
    callbacks_.push_back(cb);
  }

  // Request early stop (e.g., from callbacks).
  void request_stop() noexcept { stop_requested_ = true; }

  // Whether stop was requested.
  [[nodiscard]] bool stop_requested() const noexcept { return stop_requested_; }

  // (Optional) reset for reusing the same Trainer instance.
  void clear_stop_request() noexcept { stop_requested_ = false; }

  // Access underlying model (mutable).
  Sequential<T>& model() noexcept { return model_; }

  // Access underlying model (const).
  const Sequential<T>& model() const noexcept { return model_; }

  // Access optimizer (mutable).
  Optimizer<T>& optimizer() noexcept { return opt_; }

  // Access optimizer (const).
  const Optimizer<T>& optimizer() const noexcept { return opt_; }

  // Global step counter (increments every train batch only).
  [[nodiscard]] SizeType global_step() const noexcept { return global_step_; }

  // Used by checkpoint restore.
  void set_global_step(SizeType v) noexcept { global_step_ = v; }

  // Iterate callbacks (used by checkpoint I/O).
  template <class Fn>
  void for_each_callback(Fn&& fn) {
    for (auto* cb : callbacks_) {
      fn(*cb);
    }
  }


 private:
  // logits: [B, C], labels: [B]
  static std::pair<SizeType, SizeType> compute_accuracy_from_logits_(
      ConstTensorView<T> logits, std::span<const SizeType> labels) {
    const auto& s = logits.shape();
    if (s.rank() != 2) {
      throw std::invalid_argument("tinynn::Trainer: logits must be rank-2");
    }
    const SizeType batch = logits.row_count();
    if (static_cast<SizeType>(labels.size()) != batch) {
      throw std::invalid_argument("tinynn::Trainer: labels size != batch");
    }

    const auto pred = tinynn::argmax_by_row<T>(logits);

    SizeType correct = 0;
    for (SizeType r = 0; r < batch; ++r) {
      if (pred[static_cast<std::size_t>(r)] ==
          labels[static_cast<std::size_t>(r)]) {
        ++correct;
      }
    }
    return {correct, batch};
  }

  Sequential<T>& model_;
  SoftmaxWithLoss<T>& loss_fn_;
  Optimizer<T>& opt_;

  // Reuse vector to avoid realloc every step.
  std::vector<ParameterView<T>> params_cache_;

  std::vector<TrainerCallback<T>*> callbacks_;

  bool stop_requested_ = false;

  SizeType global_step_ = 0;

  template <class Fn>
  void for_each_cb_(Fn&& fn) {
    for (auto* cb : callbacks_) fn(*cb);
  }
};

}  // namespace tinynn
