#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include <tinynn/nn/buffer_view.h>
#include <tinynn/nn/parameter_view.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

// Train/eval phase (spec)
enum class Phase : std::uint8_t {
  Train = 0,
  Eval = 1,
};

template <class T>
class Layer {
 public:
  using ParamId = std::uintptr_t;

  virtual ~Layer() noexcept = default;

  virtual Shape output_shape(const Shape& input_shape) const = 0;
  virtual void forward(ConstTensorView<T> x, TensorView<T> y) = 0;
  virtual void backward(ConstTensorView<T> dy, TensorView<T> dx) = 0;

  // Optional parameter reset hook.
  // Default is no-op so stateless / non-trainable layers do not need to override.
  virtual void reset_parameters() {}

  // ----- phase control -----
  void set_phase(Phase p) noexcept {
    phase_.store(static_cast<std::uint8_t>(p), std::memory_order_relaxed);
  }

  [[nodiscard]] Phase phase() const noexcept {
    return static_cast<Phase>(phase_.load(std::memory_order_relaxed));
  }

  void train() noexcept { set_phase(Phase::Train); }
  void eval() noexcept { set_phase(Phase::Eval); }

  // Backward-compatible helpers
  void set_training(bool training) noexcept {
    set_phase(training ? Phase::Train : Phase::Eval);
  }

  [[nodiscard]] bool is_training() const noexcept {
    return phase() == Phase::Train;
  }

  // Trainable parameters (default: none)
  virtual void collect_parameter_views(std::vector<ParameterView<T>>& out) {
    (void)out;
  }

  // Persistent buffers for checkpointing (default: none)
  virtual void collect_buffer_views(std::vector<BufferView<T>>& out) {
    (void)out;
  }

  // Allocate a globally unique id for parameter/buffer state keys.
  static ParamId allocate_param_id() {
    return next_param_id_().fetch_add(1, std::memory_order_relaxed);
  }

  // Test/debug helper:
  // reset the global allocator so separately-constructed models in the same
  // process can receive the same deterministic ids.
  static void reset_param_id_allocator_for_testing() {
    next_param_id_().store(1, std::memory_order_relaxed);
  }

 protected:
  Layer() = default;

  // Stored phase (default train)
  std::atomic<std::uint8_t> phase_{static_cast<std::uint8_t>(Phase::Train)};

 private:
  static std::atomic<ParamId>& next_param_id_() {
    static std::atomic<ParamId> next{1};  // 0 is reserved as invalid
    return next;
  }

  Layer(const Layer&) = delete;
  Layer& operator=(const Layer&) = delete;
  Layer(Layer&&) = delete;
  Layer& operator=(Layer&&) = delete;
};

}  // namespace tinynn
