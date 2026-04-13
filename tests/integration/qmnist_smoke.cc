#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <tinynn/data/data_loader.h>
#include <tinynn/datasets/qmnist_dataset.h>
#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/layers/dropout.h>
#include <tinynn/nn/layers/relu.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/nn/sequential.h>
#include <tinynn/optim/sgd.h>
#include <tinynn/training/trainer.h>
#include <tinynn/training/callbacks/print_every.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tinynn.h>

int main() {

  if (!std::filesystem::exists("data/qmnist/qmnist-train-images-idx3-ubyte")) {
    std::cout << "[SKIP] QMNIST dataset files not found\n";
    return 0;
  }

  using T = float;

  // -----------------------
  // Paths (relative to $(SolutionDir))
  // -----------------------
  const std::string kTrainImages = "data/qmnist/qmnist-train-images-idx3-ubyte";
  const std::string kTrainLabels = "data/qmnist/qmnist-train-labels-idx2-int";
  const std::string kTestImages  = "data/qmnist/qmnist-test-images-idx3-ubyte";
  const std::string kTestLabels  = "data/qmnist/qmnist-test-labels-idx2-int";

  // -----------------------
  // Load datasets
  // -----------------------
  tinynn::QmnistDataset<T> train_ds({kTrainImages, kTrainLabels});
  tinynn::QmnistDataset<T> test_ds({kTestImages, kTestLabels});

  std::cout << "train size=" << train_ds.size()
            << " rows=" << train_ds.rows()
            << " cols=" << train_ds.cols() << "\n";
  std::cout << "test  size=" << test_ds.size()
            << " rows=" << test_ds.rows()
            << " cols=" << test_ds.cols() << "\n";

  // -----------------------
  // DataLoaders
  // -----------------------
  const tinynn::SizeType batch_train = 64;
  const tinynn::SizeType batch_eval  = 256;

  tinynn::DataLoader<T, tinynn::QmnistDataset<T>> train_loader(
      train_ds, batch_train, /*shuffle=*/true, /*seed=*/123);
  tinynn::DataLoader<T, tinynn::QmnistDataset<T>> eval_loader(
      test_ds, batch_eval, /*shuffle=*/false, /*seed=*/123);

  // -----------------------
  // Quick "read" sanity check: grab 1 batch from train_loader
  // -----------------------
  {
    auto it = train_loader.begin();
    const auto& batch = *it;

    std::cout << "batch.x shape=["
              << batch.x.shape().dim_unchecked(0) << ", "
              << batch.x.shape().dim_unchecked(1) << "]"
              << " batch.y size=" << batch.y.size() << "\n";

    // Expect [B, 784]
    if (batch.x.shape().rank() != 2 ||
        batch.x.shape().dim_unchecked(1) != train_ds.flat_dim()) {
      std::cerr << "Unexpected batch.x shape\n";
      return 1;
    }

    // pixel range sanity: [0,1]
    T mn = batch.x[0], mx = batch.x[0];
    for (tinynn::SizeType i = 0; i < batch.x.size(); ++i) {
      mn = std::min(mn, batch.x[i]);
      mx = std::max(mx, batch.x[i]);
    }
    std::cout << "pixel min=" << mn << " max=" << mx << "\n";

    // label range sanity: [0,9]
    tinynn::SizeType lmin = batch.y[0], lmax = batch.y[0];
    for (auto v : batch.y) {
      lmin = std::min(lmin, v);
      lmax = std::max(lmax, v);
    }
    std::cout << "label min=" << lmin << " max=" << lmax << "\n";

    if (mn < static_cast<T>(0) || mx > static_cast<T>(1) + static_cast<T>(1e-6)) {
      std::cerr << "Pixel range seems wrong\n";
      return 1;
    }
    if (lmax > 9) {
      std::cerr << "Label range seems wrong\n";
      return 1;
    }
  }

  // -----------------------
  // Build model: 784 -> 128 -> ReLU -> 10
  // -----------------------
  tinynn::Sequential<T> model;

  auto fc1 = std::make_unique<tinynn::Affine<T>>(train_ds.flat_dim(), 128, true);
  auto* fc1p = fc1.get();
  model.add(std::move(fc1));
  model.add(std::make_unique<tinynn::Dropout<T>>(0.5f));
  model.add(std::make_unique<tinynn::ReLU<T>>());

  auto fc2 = std::make_unique<tinynn::Affine<T>>(128, 10, true);
  auto* fc2p = fc2.get();
  model.add(std::move(fc2));

  // init (if you have these)
  fc1p->init_he_normal(123);
  fc2p->init_xavier_uniform(456);

  // -----------------------
  // Trainer
  // -----------------------
  tinynn::SoftmaxWithLoss<T> loss_fn;
  tinynn::SGD<T> sgd(/*lr=*/0.1f, /*weight_decay=*/0.0f);
  tinynn::Trainer<T> trainer(model, loss_fn, sgd);

  tinynn::FitOptions opt;
  opt.epochs = 2;      // start small

  tinynn::PrintEveryCallback<float> printer(100);
  trainer.add_callback(&printer);

  // tinynn::CheckpointCallback<float> ckpt("best.bin");
  // trainer.add_callback(&ckpt);d

  trainer.fit(train_loader, eval_loader, opt);

  // if (ckpt.has_best()) {
  //   std::cout << "Best checkpoint saved at: " << ckpt.best_path() << "\n";
  // }

  std::cout << "QMNIST smoke test OK.\n";
  return 0;
}
