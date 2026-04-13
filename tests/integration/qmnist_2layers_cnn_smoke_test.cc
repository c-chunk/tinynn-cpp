#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <tinynn/data/data_loader.h>
#include <tinynn/datasets/qmnist_dataset.h>

#include <tinynn/nn/layers/conv2d.h>
#include <tinynn/nn/layers/maxpool2d.h>
#include <tinynn/nn/layers/flatten.h>
#include <tinynn/nn/layers/relu.h>
#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/nn/sequential.h>

#include <tinynn/optim/sgd.h>
#include <tinynn/training/trainer.h>
#include <tinynn/training/callbacks/print_every.h>

#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tinynn.h>

namespace {

template <class T, class BaseLoader>
class CnnQmnistLoader {
 public:
  using SizeType = typename BaseLoader::SizeType;

  explicit CnnQmnistLoader(BaseLoader& base) : base_(&base) {}

  void set_epoch(int epoch) { base_->set_epoch(static_cast<SizeType>(epoch)); }
  void prepare_epoch() { base_->prepare_epoch(); }

  struct Iterator {
    CnnQmnistLoader* loader{};
    decltype(std::declval<BaseLoader&>().begin()) it{};
    decltype(std::declval<BaseLoader&>().end()) it_end{};

    tinynn::Batch<T> operator*() const {
      const tinynn::Batch<T> b = *it;  // copy
      return loader->to_nchw_(b);
    }

    Iterator& operator++() { ++it; return *this; }
    bool operator!=(const Iterator& /*other*/) const { return it != it_end; }
  };

  Iterator begin() { return Iterator{this, base_->begin(), base_->end()}; }
  Iterator end() { return Iterator{this, base_->end(), base_->end()}; }

 private:
  tinynn::Batch<T> to_nchw_(const tinynn::Batch<T>& b) const {
    const auto& s = b.x.shape();
    if (s.rank() != 2 || s.dim_unchecked(1) != 28 * 28) {
      throw std::invalid_argument("CnnQmnistLoader: expected base batch.x shape [B,784]");
    }
    const tinynn::SizeType B = s.dim_unchecked(0);

    tinynn::Batch<T> out;
    out.x = tinynn::Tensor<T>(tinynn::Shape{{B, 1, 28, 28}});
    std::copy(b.x.data(), b.x.data() + b.x.size(), out.x.data());
    out.y = b.y;
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

  const std::string kTrainImages = "data/qmnist/qmnist-train-images-idx3-ubyte";
  const std::string kTrainLabels = "data/qmnist/qmnist-train-labels-idx2-int";
  const std::string kTestImages  = "data/qmnist/qmnist-test-images-idx3-ubyte";
  const std::string kTestLabels  = "data/qmnist/qmnist-test-labels-idx2-int";

  tinynn::QmnistDataset<T> train_ds({kTrainImages, kTrainLabels});
  tinynn::QmnistDataset<T> test_ds({kTestImages, kTestLabels});

  std::cout << "train size=" << train_ds.size()
            << " rows=" << train_ds.rows()
            << " cols=" << train_ds.cols() << "\n";
  std::cout << "test  size=" << test_ds.size()
            << " rows=" << test_ds.rows()
            << " cols=" << test_ds.cols() << "\n";

  const tinynn::SizeType batch_train = 64;
  const tinynn::SizeType batch_eval  = 256;

  tinynn::DataLoader<T, tinynn::QmnistDataset<T>> train_base(
      train_ds, batch_train, /*shuffle=*/true, /*seed=*/123);
  tinynn::DataLoader<T, tinynn::QmnistDataset<T>> eval_base(
      test_ds, batch_eval, /*shuffle=*/false, /*seed=*/123);

  CnnQmnistLoader<T, decltype(train_base)> train_loader(train_base);
  CnnQmnistLoader<T, decltype(eval_base)>  eval_loader(eval_base);

  // ---- Model: 2-layer CNN ----
  tinynn::Sequential<T> model;

  model.add(std::make_unique<tinynn::Conv2d<T>>(tinynn::Conv2dOptions{
    .in_channels = 1,
    .out_channels = 8,
    .kernel_h = 3,
    .kernel_w = 3,
    .bias = true,
    .stride_h = 1, .stride_w = 1,
    .pad_h = 1, .pad_w = 1,
    .dilation_h = 1, .dilation_w = 1,
    .groups = 1,
  }));
  model.add(std::make_unique<tinynn::ReLU<T>>());
  model.add(std::make_unique<tinynn::MaxPool2d<T>>(tinynn::MaxPool2dOptions{})); // 2x2 stride2

  model.add(std::make_unique<tinynn::Conv2d<T>>(tinynn::Conv2dOptions{
    .in_channels = 8,
    .out_channels = 16,
    .kernel_h = 3,
    .kernel_w = 3,
    .bias = true,
    .stride_h = 1, .stride_w = 1,
    .pad_h = 1, .pad_w = 1,
    .dilation_h = 1, .dilation_w = 1,
    .groups = 1,
  }));
  model.add(std::make_unique<tinynn::ReLU<T>>());
  model.add(std::make_unique<tinynn::MaxPool2d<T>>(tinynn::MaxPool2dOptions{}));

  model.add(std::make_unique<tinynn::Flatten<T>>());
  model.add(std::make_unique<tinynn::Affine<T>>(16 * 7 * 7, 10, true));

  // ---- Trainer ----
  tinynn::SoftmaxWithLoss<T> loss_fn;

  tinynn::SGD<T> sgd(/*lr=*/0.05f, /*weight_decay=*/0.0f);

  tinynn::Trainer<T> trainer(model, loss_fn, sgd);

  tinynn::FitOptions opt;
  opt.epochs = 3;

  tinynn::PrintEveryCallback<float> printer(100);
  trainer.add_callback(&printer);

  trainer.fit(train_loader, eval_loader, opt);

  std::cout << "QMNIST 2-layer CNN smoke test OK.\n";
  return 0;
}
