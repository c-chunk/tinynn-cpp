#pragma once

#include <span>
#include <stdexcept>
#include <vector>

#include <tinynn/tensor/tensor.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

template <class T>
class SoftmaxWithLoss {
 public:
  SoftmaxWithLoss() = default;

  // Forward with index labels (recommended): labels[r] in [0, num_classes).
  // Returns mean loss over batch.
  T forward(ConstTensorView<T> logits, std::span<const SizeType> labels) {
    ensure_logits_rank2(logits);

    assert(logits.shape().rank() == 2);
    assert(labels.size() == logits.row_count());

    const SizeType batch = logits.row_count();
    const SizeType classes = logits.col_count();

    if (static_cast<SizeType>(labels.size()) != batch) {
      throw std::invalid_argument(
          "tinynn::SoftmaxWithLoss::forward: labels size != batch");
    }

    // Allocate internal buffers
    probs_ = Tensor<T>(logits.shape());
    dlogits_ = Tensor<T>(logits.shape());
    labels_.assign(labels.begin(), labels.end());

    // Compute softmax probabilities (stable) and loss
    T loss_sum = static_cast<T>(0);

    for (SizeType r = 0; r < batch; ++r) {
      const SizeType y = labels_[r];
      if (y >= classes) {
        throw std::invalid_argument(
            "tinynn::SoftmaxWithLoss::forward: label out of range");
      }

      auto in = logits.row_span(r);
      auto p = probs_.view().row_span(r);

      // max for stability
      T m = in[0];
      for (SizeType c = 1; c < classes; ++c) {
        if (in[c] > m) m = in[c];
      }

      // exp(x - m) and sum
      T sum = static_cast<T>(0);
      for (SizeType c = 0; c < classes; ++c) {
        const T e = exp_t(in[c] - m);
        p[c] = e;
        sum += e;
      }

      // normalize
      const T inv_sum = static_cast<T>(1) / sum;
      for (SizeType c = 0; c < classes; ++c) {
        p[c] *= inv_sum;
      }

      // loss: -log(p[y])
      const T py = clamp_prob(p[y]);
      loss_sum += -log_t(py);
    }

    last_batch_ = batch;
    last_classes_ = classes;
    has_labels_index_ = true;

    last_loss_ = loss_sum / static_cast<T>(batch);  // mean
    return last_loss_;
  }

  // Forward with one-hot labels: t shape [batch, classes]
  // Returns mean loss over batch.
  T forward_onehot(ConstTensorView<T> logits, ConstTensorView<T> onehot) {
    ensure_logits_rank2(logits);
    ensure_logits_rank2(onehot);

    const SizeType batch = logits.row_count();
    const SizeType classes = logits.col_count();

    if (onehot.row_count() != batch || onehot.col_count() != classes) {
      throw std::invalid_argument(
          "tinynn::SoftmaxWithLoss::forward_onehot: onehot shape mismatch");
    }

    probs_ = Tensor<T>(logits.shape());
    dlogits_ = Tensor<T>(logits.shape());
    labels_.clear();
    has_labels_index_ = false;

    T loss_sum = static_cast<T>(0);

    for (SizeType r = 0; r < batch; ++r) {
      auto in = logits.row_span(r);
      auto t = onehot.row_span(r);
      auto p = probs_.view().row_span(r);

      // max for stability
      T m = in[0];
      for (SizeType c = 1; c < classes; ++c) {
        if (in[c] > m) m = in[c];
      }

      // exp(x - m) and sum
      T sum = static_cast<T>(0);
      for (SizeType c = 0; c < classes; ++c) {
        const T e = exp_t(in[c] - m);
        p[c] = e;
        sum += e;
      }

      // normalize
      const T inv_sum = static_cast<T>(1) / sum;
      for (SizeType c = 0; c < classes; ++c) {
        p[c] *= inv_sum;
      }

      // cross-entropy: -sum_c t[c] * log(p[c])
      T row_loss = static_cast<T>(0);
      for (SizeType c = 0; c < classes; ++c) {
        if (t[c] != static_cast<T>(0)) {
          row_loss += -t[c] * log_t(clamp_prob(p[c]));
        }
      }
      loss_sum += row_loss;
    }

    last_batch_ = batch;
    last_classes_ = classes;

    last_loss_ = loss_sum / static_cast<T>(batch);  // mean
    return last_loss_;
  }

  // Backward: returns dlogits = (p - t) / batch  (mean reduction)
  // For index labels, t is implicit one-hot.
  [[nodiscard]] TensorView<T> backward() {
    ensure_forward_called();

    auto p_all = probs_.view();
    auto g_all = dlogits_.view();

    const SizeType batch = last_batch_;
    const SizeType classes = last_classes_;
    const T inv_batch = static_cast<T>(1) / static_cast<T>(batch);

    // Start with g = p.
    for (SizeType i = 0; i < g_all.size(); ++i) {
      g_all[i] = p_all[i];
    }

    if (has_labels_index_) {
      // g[r, y] -= 1
      for (SizeType r = 0; r < batch; ++r) {
        const SizeType y = labels_[r];
        g_all(r, y) -= static_cast<T>(1);
      }
    } else {
      // One-hot labels are not stored internally.
      // Use backward_onehot(onehot) for the one-hot path.
      throw std::invalid_argument(
          "tinynn::SoftmaxWithLoss::backward: onehot forward requires backward_onehot(dy) style. "
          "Use forward(logits, labels) instead.");
    }

    // mean reduction
    for (SizeType i = 0; i < g_all.size(); ++i) {
      g_all[i] *= inv_batch;
    }

    return g_all;
  }

  // For one-hot use-case, backward that accepts the onehot again:
  [[nodiscard]] TensorView<T> backward_onehot(ConstTensorView<T> onehot) {
    ensure_forward_called();
    ensure_logits_rank2(onehot);

    if (onehot.row_count() != last_batch_ || onehot.col_count() != last_classes_) {
      throw std::invalid_argument(
          "tinynn::SoftmaxWithLoss::backward_onehot: onehot shape mismatch");
    }

    auto p_all = probs_.view();
    auto g_all = dlogits_.view();
    const T inv_batch = static_cast<T>(1) / static_cast<T>(last_batch_);

    for (SizeType i = 0; i < g_all.size(); ++i) {
      g_all[i] = p_all[i];
    }

    for (SizeType r = 0; r < last_batch_; ++r) {
      auto t = onehot.row_span(r);
      auto g = g_all.row_span(r);
      for (SizeType c = 0; c < last_classes_; ++c) {
        g[c] -= t[c];
      }
    }

    for (SizeType i = 0; i < g_all.size(); ++i) {
      g_all[i] *= inv_batch;
    }

    return g_all;
  }

  // Accessors
  T last_loss() const noexcept { return last_loss_; }

  [[nodiscard]] ConstTensorView<T> probs() const noexcept { return probs_.view(); }
  [[nodiscard]] TensorView<T> dlogits() noexcept { return dlogits_.view(); }

 private:
  static void ensure_logits_rank2(ConstTensorView<T> x) {
    if (x.shape().rank() != 2) {
      throw std::invalid_argument(
          "tinynn::SoftmaxWithLoss: input must be rank-2");
    }
  }

  void ensure_forward_called() const {
    if (last_batch_ == 0 || last_classes_ == 0 || probs_.empty()) {
      throw std::invalid_argument(
          "tinynn::SoftmaxWithLoss: call forward() first");
    }
  }

  static T exp_t(T x) {
    using std::exp;
    return static_cast<T>(exp(static_cast<double>(x)));
  }

  static T log_t(T x) {
    using std::log;
    return static_cast<T>(log(static_cast<double>(x)));
  }

  static T clamp_prob(T p) {
    // Avoid log(0)
    const T eps = static_cast<T>(1e-12);
    if (p < eps) return eps;
    return p;
  }

  Tensor<T> probs_{};
  Tensor<T> dlogits_{};

  std::vector<SizeType> labels_{};
  bool has_labels_index_ = false;

  SizeType last_batch_ = 0;
  SizeType last_classes_ = 0;
  T last_loss_ = static_cast<T>(0);
};

}  // namespace tinynn
