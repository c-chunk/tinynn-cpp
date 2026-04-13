// full_checkpoint_roundtrip_batch2d_test.cc
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tinynn/nn/sequential.h>
#include <tinynn/nn/layer.h>
#include <tinynn/nn/layers/conv2d.h>
#include <tinynn/nn/layers/batch_norm2d.h>
#include <tinynn/nn/layers/relu.h>
#include <tinynn/nn/layers/flatten.h>
#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/optim/adam.h>
#include <tinynn/training/trainer.h>
#include <tinynn/training/callback.h>
#include <tinynn/training/callbacks/checkpoint_state_io.h>

namespace {

template <class T>
struct Batch {
  tinynn::Tensor<T> x;
  std::vector<tinynn::SizeType> y;
};

template <class T>
class FixedLoader {
 public:
  explicit FixedLoader(std::vector<Batch<T>> batches)
      : batches_(std::move(batches)) {}

  void set_epoch(int) {}
  void prepare_epoch() {}

  auto begin() const { return batches_.begin(); }
  auto end() const { return batches_.end(); }

 private:
  std::vector<Batch<T>> batches_;
};

template <class T>
class StepCounterCallback final : public tinynn::TrainerCallback<T> {
 public:
  const char* checkpoint_tag() const noexcept override {
    return "StepCounter";
  }

  void on_train_step_end(tinynn::Trainer<T>&, int, tinynn::SizeType,
                         const tinynn::StepResult<T>&) override {
    ++count_;
  }

  void save_state(tinynn::CheckpointWriter& w) const override {
    w.write_u64(static_cast<uint64_t>(count_));
  }

  void load_state(tinynn::CheckpointReader& r) override {
    count_ = static_cast<std::uint64_t>(r.read_u64());
  }

  std::uint64_t count() const noexcept { return count_; }

 private:
  std::uint64_t count_ = 0;
};

template <class T>
std::vector<T> snapshot_params(tinynn::Sequential<T>& model) {
  std::vector<tinynn::ParameterView<T>> pvs;
  model.collect_parameter_views(pvs);

  std::vector<T> flat;
  for (auto& pv : pvs) {
    auto p = pv.param;
    for (tinynn::SizeType i = 0; i < p.size(); ++i) {
      flat.push_back(p[i]);
    }
  }
  return flat;
}

template <class T>
std::vector<T> snapshot_buffers(tinynn::Sequential<T>& model) {
  std::vector<tinynn::BufferView<T>> bvs;
  model.collect_buffer_views(bvs);

  std::vector<T> flat;
  for (auto& bv : bvs) {
    auto b = bv.buf;
    for (tinynn::SizeType i = 0; i < b.size(); ++i) {
      flat.push_back(b[i]);
    }
  }
  return flat;
}

template <class T>
void assert_exact_equal(const std::vector<T>& a, const std::vector<T>& b,
                        const char* what) {
  if (a.size() != b.size()) {
    throw std::runtime_error(std::string("[NG] ") + what +
                             ": size mismatch");
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) {
      throw std::runtime_error(std::string("[NG] ") + what +
                               ": mismatch at i=" + std::to_string(i));
    }
  }
}

}  // namespace

int main() {
  using T = double;

  try {
    // Small deterministic NCHW dataset
    // B=4, C=1, H=W=4
    const tinynn::SizeType B = 4;
    const tinynn::SizeType C = 1;
    const tinynn::SizeType H = 4;
    const tinynn::SizeType W = 4;
    const tinynn::SizeType num_classes = 2;

    Batch<T> b0{
        tinynn::Tensor<T>({B, C, H, W}),
        std::vector<tinynn::SizeType>(B),
    };

    // sample 0: low values
    {
      T v = 0.0;
      for (tinynn::SizeType h = 0; h < H; ++h) {
        for (tinynn::SizeType w = 0; w < W; ++w) {
          b0.x(0, 0, h, w) = v;
          v += 0.1;
        }
      }
      b0.y[0] = 0;
    }

    // sample 1: shifted higher
    {
      T v = 1.0;
      for (tinynn::SizeType h = 0; h < H; ++h) {
        for (tinynn::SizeType w = 0; w < W; ++w) {
          b0.x(1, 0, h, w) = v;
          v += 0.1;
        }
      }
      b0.y[1] = 1;
    }

    // sample 2: checkerboard-ish
    {
      for (tinynn::SizeType h = 0; h < H; ++h) {
        for (tinynn::SizeType w = 0; w < W; ++w) {
          b0.x(2, 0, h, w) =
              ((h + w) % 2 == 0) ? static_cast<T>(0.5) : static_cast<T>(1.5);
        }
      }
      b0.y[2] = 1;
    }

    // sample 3: inverse checkerboard-ish
    {
      for (tinynn::SizeType h = 0; h < H; ++h) {
        for (tinynn::SizeType w = 0; w < W; ++w) {
          b0.x(3, 0, h, w) =
              ((h + w) % 2 == 0) ? static_cast<T>(1.5) : static_cast<T>(0.5);
        }
      }
      b0.y[3] = 0;
    }

    FixedLoader<T> loader({b0});

    // Conv(1->2, 3x3, pad1) => [B,2,4,4]
    // BN2d(2)
    // ReLU
    // Flatten => [B, 2*4*4]
    // Affine => [B,2]
    const tinynn::SizeType conv_out_channels = 2;
    const tinynn::SizeType flat_dim = conv_out_channels * H * W;

    // -------- Trainer A (reference) --------
    tinynn::Layer<T>::reset_param_id_allocator_for_testing();

    tinynn::Sequential<T> modelA;
    modelA.add(std::make_unique<tinynn::Conv2d<T>>(tinynn::Conv2dOptions{
        .in_channels = 1,
        .out_channels = conv_out_channels,
        .kernel_h = 3,
        .kernel_w = 3,
        .bias = true,
        .stride_h = 1,
        .stride_w = 1,
        .pad_h = 1,
        .pad_w = 1,
        .dilation_h = 1,
        .dilation_w = 1,
        .groups = 1,
    }));
    modelA.add(std::make_unique<tinynn::BatchNorm2d<T>>(conv_out_channels));
    modelA.add(std::make_unique<tinynn::ReLU<T>>());
    modelA.add(std::make_unique<tinynn::Flatten<T>>());
    modelA.add(std::make_unique<tinynn::Affine<T>>(flat_dim, num_classes));

    modelA.for_each_layer([&](tinynn::Layer<T>& lyr) {
      if (auto* a = dynamic_cast<tinynn::Affine<T>*>(&lyr)) {
        a->init_he_normal(123);
      }
    });

    tinynn::SoftmaxWithLoss<T> lossA;
    tinynn::Adam<T> optA(
        /*lr=*/static_cast<T>(1e-2),
        /*beta1=*/static_cast<T>(0.9),
        /*beta2=*/static_cast<T>(0.999),
        /*eps=*/static_cast<T>(1e-8),
        /*weight_decay=*/static_cast<T>(1e-4));

    tinynn::Trainer<T> trainerA(modelA, lossA, optA);
    StepCounterCallback<T> cbA;
    trainerA.add_callback(&cbA);

    for (int e = 0; e < 2; ++e) {
      (void)trainerA.train_epoch(loader, e);
      (void)trainerA.eval_epoch(loader, e);
    }

    const std::string path = "full_ckpt_roundtrip_bn2d.bin";
    const int next_epoch = 2;
    tinynn::save_full_checkpoint<T>(path, trainerA, next_epoch);

    for (int e = 2; e < 5; ++e) {
      (void)trainerA.train_epoch(loader, e);
      (void)trainerA.eval_epoch(loader, e);
    }

    const auto ref_params = snapshot_params(modelA);
    const auto ref_buffers = snapshot_buffers(modelA);
    const auto ref_global_step = trainerA.global_step();
    const auto ref_cb_count = cbA.count();

    // -------- Trainer B (restore) --------
    tinynn::Layer<T>::reset_param_id_allocator_for_testing();

    tinynn::Sequential<T> modelB;
    modelB.add(std::make_unique<tinynn::Conv2d<T>>(tinynn::Conv2dOptions{
        .in_channels = 1,
        .out_channels = conv_out_channels,
        .kernel_h = 3,
        .kernel_w = 3,
        .bias = true,
        .stride_h = 1,
        .stride_w = 1,
        .pad_h = 1,
        .pad_w = 1,
        .dilation_h = 1,
        .dilation_w = 1,
        .groups = 1,
    }));
    modelB.add(std::make_unique<tinynn::BatchNorm2d<T>>(conv_out_channels));
    modelB.add(std::make_unique<tinynn::ReLU<T>>());
    modelB.add(std::make_unique<tinynn::Flatten<T>>());
    modelB.add(std::make_unique<tinynn::Affine<T>>(flat_dim, num_classes));

    modelB.for_each_layer([&](tinynn::Layer<T>& lyr) {
      if (auto* a = dynamic_cast<tinynn::Affine<T>*>(&lyr)) {
        a->init_he_normal(999);  // intentionally different
      }
    });

    tinynn::SoftmaxWithLoss<T> lossB;
    tinynn::Adam<T> optB(
        /*lr=*/static_cast<T>(1e-1),
        /*beta1=*/static_cast<T>(0.5),
        /*beta2=*/static_cast<T>(0.5),
        /*eps=*/static_cast<T>(1e-3),
        /*weight_decay=*/static_cast<T>(0));  // intentionally different

    tinynn::Trainer<T> trainerB(modelB, lossB, optB);
    StepCounterCallback<T> cbB;
    trainerB.add_callback(&cbB);

    int restored_next_epoch = -1;
    tinynn::load_full_checkpoint<T>(path, trainerB, &restored_next_epoch);

    if (restored_next_epoch != next_epoch) {
      throw std::runtime_error("[NG] next_epoch mismatch");
    }

    for (int e = restored_next_epoch; e < 5; ++e) {
      (void)trainerB.train_epoch(loader, e);
      (void)trainerB.eval_epoch(loader, e);
    }

    const auto got_params = snapshot_params(modelB);
    const auto got_buffers = snapshot_buffers(modelB);
    const auto got_global_step = trainerB.global_step();
    const auto got_cb_count = cbB.count();

    assert_exact_equal(ref_params, got_params, "full checkpoint params");
    assert_exact_equal(ref_buffers, got_buffers, "full checkpoint buffers");

    if (ref_global_step != got_global_step) {
      throw std::runtime_error("[NG] global_step mismatch");
    }
    if (ref_cb_count != got_cb_count) {
      throw std::runtime_error("[NG] callback counter mismatch");
    }

    std::cout
        << "[OK] full checkpoint BN2d roundtrip matched "
           "(params/buffers/global_step/callback)\n";
    std::cout << "full_checkpoint_roundtrip_batch2d_test passed.\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
