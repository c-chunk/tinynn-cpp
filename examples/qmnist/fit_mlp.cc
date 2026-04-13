#include <iostream>
#include <memory>
#include <span>
#include <vector>

#include <tinynn/data/batch.h>
#include <tinynn/data/data_loader.h>
#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/layers/relu.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/nn/sequential.h>
#include <tinynn/optim/sgd.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tinynn.h>
#include <tinynn/training/trainer.h>

namespace {

// A tiny in-memory dataset for multi-class classification.
// x shape: [N, 3], y shape: [N]
template <class T>
class ToyDataset {
 public:
  ToyDataset() {
    // 6 samples, 2 classes
    // class 0
    x_ = {
        static_cast<T>(0),
        static_cast<T>(0),
        static_cast<T>(1),
        static_cast<T>(0),
        static_cast<T>(1),
        static_cast<T>(0),
        static_cast<T>(1),
        static_cast<T>(0),
        static_cast<T>(0),

        // class 1
        static_cast<T>(1),
        static_cast<T>(1),
        static_cast<T>(0),
        static_cast<T>(1),
        static_cast<T>(1),
        static_cast<T>(1),
        static_cast<T>(0),
        static_cast<T>(1),
        static_cast<T>(1),
    };

    y_ = {
        0, 0, 0, 1, 1, 1,
    };
  }

  tinynn::SizeType size() const noexcept {
    return static_cast<tinynn::SizeType>(y_.size());
  }

  void get_batch(std::span<const tinynn::SizeType> indices,
                 tinynn::Batch<T>* out) const {
    if (!out) {
      throw std::invalid_argument("ToyDataset::get_batch: out is null");
    }

    const tinynn::SizeType batch_size =
        static_cast<tinynn::SizeType>(indices.size());

    out->x = tinynn::Tensor<T>(
        tinynn::Shape{batch_size, static_cast<tinynn::SizeType>(3)});
    out->y.resize(batch_size);

    auto xv = out->x.view();

    for (tinynn::SizeType i = 0; i < batch_size; ++i) {
      const tinynn::SizeType idx = indices[i];
      const std::size_t base = static_cast<std::size_t>(idx) * 3;

      xv(i, 0) = x_[base + 0];
      xv(i, 1) = x_[base + 1];
      xv(i, 2) = x_[base + 2];
      out->y[i] = y_[static_cast<std::size_t>(idx)];
    }
  }

 private:
  std::vector<T> x_;
  std::vector<tinynn::SizeType> y_;
};

}  // namespace

int main() {
  using T = float;

  try {
    // -----------------------------
    // Dataset / DataLoader
    // -----------------------------
    ToyDataset<T> train_ds;
    ToyDataset<T> eval_ds;

    tinynn::DataLoader<T, ToyDataset<T>> train_loader(
        train_ds, /*batch_size=*/3, /*shuffle=*/true, /*seed=*/123);

    tinynn::DataLoader<T, ToyDataset<T>> eval_loader(
        eval_ds, /*batch_size=*/3, /*shuffle=*/false, /*seed=*/123);

    // -----------------------------
    // Model: 3 -> 8 -> ReLU -> 2
    // -----------------------------
    tinynn::Sequential<T> model;

    auto fc1 = std::make_unique<tinynn::Affine<T>>(3, 8, true);
    auto* fc1p = fc1.get();
    model.add(std::move(fc1));

    model.add(std::make_unique<tinynn::ReLU<T>>());

    auto fc2 = std::make_unique<tinynn::Affine<T>>(8, 2, true);
    auto* fc2p = fc2.get();
    model.add(std::move(fc2));

    // Recommended deterministic init for reproducibility
    fc1p->init_he_normal(123);
    fc2p->init_xavier_uniform(456);

    // -----------------------------
    // Loss / Optimizer / Trainer
    // -----------------------------
    tinynn::SoftmaxWithLoss<T> loss_fn;
    tinynn::SGD<T> optimizer(/*lr=*/static_cast<T>(0.1),
                             /*weight_decay=*/static_cast<T>(0.0));

    tinynn::Trainer<T> trainer(model, loss_fn, optimizer);

    // -----------------------------
    // Fit
    // -----------------------------
    tinynn::FitOptions opt;
    opt.epochs = 30;

    std::cout << "Start MLP training example..." << std::endl;
    trainer.fit(train_loader, eval_loader, opt);
    std::cout << "Training finished." << std::endl;

    // -----------------------------
    // Final evaluation
    // -----------------------------
    const auto ev = trainer.eval_epoch(eval_loader, /*epoch=*/opt.epochs - 1);

    std::cout << "Final eval loss=" << ev.loss << " acc=" << ev.accuracy << " ("
              << ev.num_correct << "/" << ev.num_samples << ")\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
