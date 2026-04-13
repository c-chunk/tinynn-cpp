#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <tinynn/data/data_loader.h>
#include <tinynn/datasets/qmnist_dataset.h>

#include <tinynn/nn/layers/conv2d.h>
#include <tinynn/nn/layers/flatten.h>
#include <tinynn/nn/layers/relu.h>
#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/layers/maxpool2d.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/nn/sequential.h>

#include <tinynn/optim/sgd.h>
#include <tinynn/training/trainer.h>
#include <tinynn/training/callbacks/print_every.h>
// #include <tinynn/training/callbacks/checkpoint.h>

#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tinynn.h>

namespace {

// ------------------------------------------------------------
// Loader wrapper: converts Batch.x from [B,784] -> [B,1,28,28].
// Keeps labels as-is.
// ------------------------------------------------------------
template <class T, class BaseLoader>
class CnnQmnistLoader {
 public:
  using SizeType = typename BaseLoader::SizeType;  // std::size_t

  explicit CnnQmnistLoader(BaseLoader& base) : base_(&base) {}

  // Trainer hooks (detected by SFINAE in tinynn::detail)
  void set_epoch(int epoch) { base_->set_epoch(static_cast<SizeType>(epoch)); }
  void prepare_epoch() { base_->prepare_epoch(); }

  struct Iterator {
    CnnQmnistLoader* loader{};
    decltype(std::declval<BaseLoader&>().begin()) it{};
    decltype(std::declval<BaseLoader&>().end()) it_end{};

    tinynn::Batch<T> operator*() const {
      const tinynn::Batch<T> b = *it;  // base batch (copied)
      return loader->to_nchw_(b);
    }

    Iterator& operator++() {
      ++it;
      return *this;
    }

    bool operator!=(const Iterator& /*other*/) const {
      return it != it_end;
    }
  };

  Iterator begin() {
    return Iterator{this, base_->begin(), base_->end()};
  }

  Iterator end() {
    return Iterator{this, base_->end(), base_->end()};
  }

 private:
  tinynn::Batch<T> to_nchw_(const tinynn::Batch<T>& b) const {
    // Expect base.x: [B,784]
    const auto& s = b.x.shape();
    if (s.rank() != 2 || s.dim_unchecked(1) != 28 * 28) {
      throw std::invalid_argument(
          "CnnQmnistLoader: expected base batch.x shape [B,784]");
    }
    const tinynn::SizeType B = s.dim_unchecked(0);

    tinynn::Batch<T> out;
    out.x = tinynn::Tensor<T>(tinynn::Shape{{B, 1, 28, 28}});
    std::copy(b.x.data(), b.x.data() + b.x.size(), out.x.data());

    out.y = b.y;  // copy labels (vector)
    return out;
  }

  BaseLoader* base_ = nullptr;
};

}  // namespace

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
  // Base DataLoaders (produce [B,784])
  // -----------------------
  const tinynn::SizeType batch_train = 64;
  const tinynn::SizeType batch_eval  = 256;

  tinynn::DataLoader<T, tinynn::QmnistDataset<T>> train_base(
      train_ds, batch_train, /*shuffle=*/true, /*seed=*/123);
  tinynn::DataLoader<T, tinynn::QmnistDataset<T>> eval_base(
      test_ds, batch_eval, /*shuffle=*/false, /*seed=*/123);

  // Wrapped loaders for CNN (produce [B,1,28,28])
  CnnQmnistLoader<T, decltype(train_base)> train_loader(train_base);
  CnnQmnistLoader<T, decltype(eval_base)> eval_loader(eval_base);

  // -----------------------
  // Sanity check: one batch
  // -----------------------
  {
    auto it = train_loader.begin();
    const auto batch = *it;  // Batch<T> (by value)

    std::cout << "cnn batch.x shape=["
              << batch.x.shape().dim_unchecked(0) << ", "
              << batch.x.shape().dim_unchecked(1) << ", "
              << batch.x.shape().dim_unchecked(2) << ", "
              << batch.x.shape().dim_unchecked(3) << "]"
              << " batch.y size=" << batch.y.size() << "\n";

    if (batch.x.shape().rank() != 4 ||
        batch.x.shape().dim_unchecked(1) != 1 ||
        batch.x.shape().dim_unchecked(2) != 28 ||
        batch.x.shape().dim_unchecked(3) != 28) {
      std::cerr << "Unexpected cnn batch.x shape\n";
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
  }

  // -----------------------
  // Build model (minimal CNN):
  //   [B,1,28,28]
  // -> Conv2d(1->8,k3) => [B,8,26,26]
  // -> ReLU            => [B,8,26,26]
  // -> Flatten         => [B,5408]
  // -> Affine          => [B,10]
  // -----------------------
  tinynn::Sequential<T> model;

  model.add(std::make_unique<tinynn::Conv2d<T>>(tinynn::Conv2dOptions{
      .in_channels = 1,
      .out_channels = 8,
      .kernel_h = 3,
      .kernel_w = 3,
      .bias = true,
      .pad_h = 1,
      .pad_w = 1,
  }));
  model.add(std::make_unique<tinynn::ReLU<T>>());
  model.add(std::make_unique<tinynn::MaxPool2d<T>>(tinynn::MaxPool2dOptions{}));
  model.add(std::make_unique<tinynn::Flatten<T>>());
  model.add(std::make_unique<tinynn::Affine<T>>(8 * 14 * 14, 10, true));

  // -----------------------
  // Trainer
  // -----------------------
  tinynn::SoftmaxWithLoss<T> loss_fn;
  tinynn::SGD<T> sgd(/*lr=*/0.05f, /*weight_decay=*/0.0f);
  tinynn::Trainer<T> trainer(model, loss_fn, sgd);

  tinynn::FitOptions opt;
  opt.epochs = 2;

  tinynn::PrintEveryCallback<float> printer(100);
  trainer.add_callback(&printer);

  // tinynn::CheckpointCallback<float> ckpt("cnn_best.bin");
  // trainer.add_callback(&ckpt);

  trainer.fit(train_loader, eval_loader, opt);

  std::cout << "QMNIST CNN smoke test OK.\n";
  return 0;
}
