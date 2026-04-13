#pragma once

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tinynn/nn/buffer_view.h>
#include <tinynn/nn/layer.h>
#include <tinynn/nn/parameter_view.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor.h>

namespace tinynn {

template <class T>
class Sequential {
 public:
  Sequential() = default;

  void add(std::unique_ptr<Layer<T>> layer) {
    if (!layer) {
      throw std::invalid_argument("tinynn::Sequential::add: layer is null");
    }
    // New layer inherits current model phase.
    layer->set_phase(phase_);
    layers_.push_back(std::move(layer));
    // Shapes/buffers are built lazily in build()/forward().
    built_ = false;
  }

  // Reset parameters/buffers of all layers in forward order.
  // Stateless layers are free to keep the default no-op implementation.
  void reset_parameters() {
    for (auto& lyr : layers_) {
      lyr->reset_parameters();
    }
    // Cached activations / gradients were built for previous parameter state.
    // Rebuild lazily on next forward().
    built_ = false;
  }

  // ----- mode control -----
  void train() noexcept { set_phase_(Phase::Train); }
  void eval() noexcept { set_phase_(Phase::Eval); }

  [[nodiscard]] bool is_training() const noexcept {
    return phase_ == Phase::Train;
  }

  [[nodiscard]] Phase phase() const noexcept { return phase_; }

  [[nodiscard]] SizeType layer_count() const noexcept {
    return static_cast<SizeType>(layers_.size());
  }

  // Build internal buffers for a given input shape.
  void build(const Shape& input_shape) {
    if (layers_.empty()) {
      throw std::invalid_argument("tinynn::Sequential::build: no layers");
    }

    shapes_.clear();
    shapes_.reserve(layers_.size() + 1);
    shapes_.push_back(input_shape);

    // Infer shapes.
    for (const auto& lyr : layers_) {
      const Shape out = lyr->output_shape(shapes_.back());
      shapes_.push_back(out);
    }

    // Allocate activations (outputs of each layer).
    // acts_[0] is not owned here (input comes from caller),
    // so we allocate acts_[1..L].
    acts_.clear();
    acts_.resize(layers_.size() + 1);
    for (SizeType i = 1; i < static_cast<SizeType>(acts_.size()); ++i) {
      acts_[i] = Tensor<T>(shapes_[i]);
    }

    // Allocate gradients for backward:
    // grads_[0..L-1] are owned; grads_[L] is dy from caller (not owned).
    grads_.clear();
    grads_.resize(layers_.size() + 1);
    for (SizeType i = 0; i < static_cast<SizeType>(layers_.size()); ++i) {
      grads_[i] = Tensor<T>(shapes_[i]);
    }

    built_ = true;
  }

  // Forward pass.
  // - x: input view with shape == shapes_[0] (or build will be called)
  // Returns a view of the final activation.
  [[nodiscard]] TensorView<T> forward(ConstTensorView<T> x) {
    if (layers_.empty()) {
      throw std::invalid_argument("tinynn::Sequential::forward: no layers");
    }

    if (!built_) {
      build(x.shape());
    } else {
      // Current simple policy:
      // rebuild only when total volume changes.
      // (This preserves the current behavior of your existing Sequential.)
      if (x.shape().volume() != shapes_.front().volume()) {
        build(x.shape());
      }
    }

    ConstTensorView<T> cur_in = x;

    for (SizeType i = 0; i < static_cast<SizeType>(layers_.size()); ++i) {
      TensorView<T> cur_out = acts_[i + 1].view();
      layers_[i]->forward(cur_in, cur_out);
      cur_in = cur_out;  // implicit conversion to const view
    }

    return acts_.back().view();
  }

  // Backward pass.
  // - dy: gradient w.r.t final output (same shape as last activation)
  // Returns a view of gradient w.r.t input (dx0).
  [[nodiscard]] TensorView<T> backward(ConstTensorView<T> dy) {
    if (!built_) {
      throw std::invalid_argument(
          "tinynn::Sequential::backward: call forward/build first");
    }
    if (layers_.empty()) {
      throw std::invalid_argument("tinynn::Sequential::backward: no layers");
    }

    // Minimal validation (keep existing policy).
    if (dy.size() != shapes_.back().volume()) {
      throw std::invalid_argument(
          "tinynn::Sequential::backward: dy shape mismatch");
    }

    ConstTensorView<T> cur_dy = dy;

    for (SizeType rev = 0; rev < static_cast<SizeType>(layers_.size()); ++rev) {
      const SizeType i = static_cast<SizeType>(layers_.size() - 1 - rev);

      // dx buffer for this layer is grads_[i] (shape of layer input).
      TensorView<T> cur_dx = grads_[i].view();
      layers_[i]->backward(cur_dy, cur_dx);

      // For next layer (previous in chain), dy becomes current dx.
      cur_dy = cur_dx;
    }

    return grads_.front().view();
  }

  // Iterate layers in forward order.
  // Useful for reseeding, diagnostics, or custom inspection.
  template <class Fn>
  void for_each_layer(Fn&& fn) {
    for (auto& p : layers_) {
      fn(*p);
    }
  }

  template <class Fn>
  void for_each_layer(Fn&& fn) const {
    for (const auto& p : layers_) {
      fn(*p);
    }
  }

  // Collect all parameter views (for optimizer / checkpoint).
  void collect_parameter_views(std::vector<ParameterView<T>>& out) {
    out.clear();

    for (auto& lyr : layers_) {
      lyr->collect_parameter_views(out);
    }

    uintptr_t next_id = 1;
    for (auto& pv : out) {
      pv.id = next_id++;
    }
  }

  // Collect persistent buffers (e.g. BatchNorm running stats).
  void collect_buffer_views(std::vector<BufferView<T>>& out) {
    out.clear();

    for (auto& lyr : layers_) {
      lyr->collect_buffer_views(out);
    }

    // Count parameters first so buffer ids do not overlap with parameter ids.
    std::vector<ParameterView<T>> params_tmp;
    for (auto& lyr : layers_) {
      lyr->collect_parameter_views(params_tmp);
    }

    uintptr_t next_id = static_cast<uintptr_t>(params_tmp.size()) + 1;
    for (auto& bv : out) {
      bv.id = next_id++;
    }
  }

  // Access last forward output (after forward).
  [[nodiscard]] TensorView<T> output() {
    if (!built_) {
      throw std::invalid_argument("tinynn::Sequential::output: not built");
    }
    return acts_.back().view();
  }

 private:
  void set_phase_(Phase p) noexcept {
    phase_ = p;
    for (auto& lyr : layers_) {
      lyr->set_phase(p);
    }
  }

 private:
  Phase phase_ = Phase::Train;
  std::vector<std::unique_ptr<Layer<T>>> layers_;

  bool built_ = false;
  std::vector<Shape> shapes_;

  // activations: acts_[1..L] are owned buffers, acts_[0] is caller-owned input
  std::vector<Tensor<T>> acts_;

  // gradients: grads_[0..L-1] are owned buffers, final dy is caller-owned
  std::vector<Tensor<T>> grads_;
};

}  // namespace tinynn
