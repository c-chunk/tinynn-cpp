#include <iostream>
#include <memory>
#include <span>
#include <vector>

#include <tinynn/data/batch.h>
#include <tinynn/data/data_loader.h>
#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/layers/relu.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/nn/predict.h>
#include <tinynn/nn/sequential.h>
#include <tinynn/optim/sgd.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tinynn.h>
#include <tinynn/training/trainer.h>

namespace {

// A tiny in-memory dataset for binary classification.
// x shape: [N, 2], y shape: [N]
template <class T>
class ToyDataset {
 public:
  ToyDataset() {
    // Two simple clusters:
    // class 0: lower-left
    // class 1: upper-right
    x_ = {
        static_cast<T>(0.0), static_cast<T>(0.0),  // 0
        static_cast<T>(0.0), static_cast<T>(1.0),  // 0
        static_cast<T>(1.0), static_cast<T>(0.0),  // 0
        static_cast<T>(1.0), static_cast<T>(1.0),  // 1
        static_cast<T>(2.0), static_cast<T>(1.0),  // 1
        static_cast<T>(1.0), static_cast<T>(2.0),  // 1
        static_cast<T>(2.0), static_cast<T>(2.0),  // 1
        static_cast<T>(2.0), static_cast<T>(0.0),  // 0
    };

    y_ = {
        0, 0, 0, 1, 1, 1, 1, 0,
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
        tinynn::Shape{batch_size, static_cast<tinynn::SizeType>(2)});
    out->y.resize(batch_size);

    auto xv = out->x.view();
    for (tinynn::SizeType i = 0; i < batch_size; ++i) {
      const tinynn::SizeType idx = indices[i];
      xv(i, 0) = x_[static_cast<std::size_t>(idx) * 2 + 0];
      xv(i, 1) = x_[static_cast<std::size_t>(idx) * 2 + 1];
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
        train_ds, /*batch_size=*/4, /*shuffle=*/true, /*seed=*/123);

    tinynn::DataLoader<T, ToyDataset<T>> eval_loader(
        eval_ds, /*batch_size=*/4, /*shuffle=*/false, /*seed=*/123);

    // -----------------------------
    // Model: 2 -> 8 -> ReLU -> 2
    // -----------------------------
    tinynn::Sequential<T> model;

    auto fc1 = std::make_unique<tinynn::Affine<T>>(2, 8, true);
    auto* fc1p = fc1.get();
    model.add(std::move(fc1));

    model.add(std::make_unique<tinynn::ReLU<T>>());

    auto fc2 = std::make_unique<tinynn::Affine<T>>(8, 2, true);
    auto* fc2p = fc2.get();
    model.add(std::move(fc2));

    // Deterministic initialization for reproducibility
    fc1p->init_he_normal(123);
    fc2p->init_xavier_uniform(456);

    // -----------------------------
    // Loss / Optimizer / Trainer
    // -----------------------------
    tinynn::SoftmaxWithLoss<T> loss_fn;
    tinynn::SGD<T> optimizer(static_cast<T>(0.1), static_cast<T>(0.0));
    tinynn::Trainer<T> trainer(model, loss_fn, optimizer);

    tinynn::FitOptions opt;
    opt.epochs = 50;

    std::cout << "Start training tiny classifier..." << std::endl;
    trainer.fit(train_loader, eval_loader, opt);
    std::cout << "Training finished." << std::endl;

    // -----------------------------
    // Inference on the full dataset
    // -----------------------------
    tinynn::Batch<T> full_batch;
    {
      std::vector<tinynn::SizeType> all_indices(train_ds.size());
      for (tinynn::SizeType i = 0; i < train_ds.size(); ++i) {
        all_indices[static_cast<std::size_t>(i)] = i;
      }
      train_ds.get_batch(std::span<const tinynn::SizeType>(all_indices.data(),
                                                           all_indices.size()),
                         &full_batch);
    }

    auto pred = tinynn::predict_label<T>(model, full_batch.x.view());

    std::cout << "\nPredictions:\n";
    for (tinynn::SizeType i = 0; i < full_batch.batch_size(); ++i) {
      std::cout << "  sample " << i
                << "  label=" << full_batch.y[static_cast<std::size_t>(i)]
                << "  pred=" << pred[static_cast<std::size_t>(i)] << '\n';
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
