#include <iostream>
#include <memory>
#include <vector>

#include <tinynn/data/data_loader.h>
#include <tinynn/data/dummy_dataset.h>
#include <tinynn/nn/sequential.h>
#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/layers/relu.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/optim/sgd.h>
#include <tinynn/training/trainer.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tinynn.h>

int main() {
  using T = float;

  // -----------------------
  // Build model: 3 -> 4 -> ReLU -> 2
  // -----------------------
  tinynn::Sequential<T> model;

  auto fc1 = std::make_unique<tinynn::Affine<T>>(3, 4, true);
  auto* fc1p = fc1.get();
  model.add(std::move(fc1));

  model.add(std::make_unique<tinynn::ReLU<T>>());

  auto fc2 = std::make_unique<tinynn::Affine<T>>(4, 2, true);
  auto* fc2p = fc2.get();
  model.add(std::move(fc2));

  // init
  fc1p->init_he_normal(123);
  fc2p->init_xavier_uniform(456);

  // -----------------------
  // Dummy dataset (same 4 samples as before)
  // -----------------------
  tinynn::Tensor<T> x_all(tinynn::Shape{4, 3});
  x_all(0,0)=0; x_all(0,1)=0; x_all(0,2)=1;
  x_all(1,0)=0; x_all(1,1)=1; x_all(1,2)=0;
  x_all(2,0)=1; x_all(2,1)=0; x_all(2,2)=0;
  x_all(3,0)=1; x_all(3,1)=1; x_all(3,2)=1;

  std::vector<tinynn::SizeType> y_all = {0, 1, 1, 0};

  tinynn::DummyDataset<T> dataset(std::move(x_all), std::move(y_all));

  // -----------------------
  // DataLoaders
  // - train: batch_size=2, shuffle=true
  // - eval : batch_size=4, shuffle=false
  // -----------------------
  tinynn::DataLoader<T, tinynn::DummyDataset<T>> train_loader(dataset, /*batch_size=*/2,
                                                              /*shuffle=*/true, /*seed=*/123);
  tinynn::DataLoader<T, tinynn::DummyDataset<T>> eval_loader(dataset, /*batch_size=*/4,
                                                             /*shuffle=*/false, /*seed=*/123);

  // -----------------------
  // Trainer
  // -----------------------
  tinynn::SoftmaxWithLoss<T> loss_fn;
  tinynn::SGD<T> sgd(/*lr=*/0.1f, /*weight_decay=*/0.0f);
  tinynn::Trainer<T> trainer(model, loss_fn, sgd);

  tinynn::FitOptions opt;
  opt.epochs = 20;

  trainer.fit(train_loader, eval_loader, opt);

  std::cout << "DummyDataset + DataLoader smoke test OK.\n";
  return 0;
}
