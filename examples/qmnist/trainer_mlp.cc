#include <iostream>
#include <memory>
#include <span>

#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/layers/relu.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/nn/sequential.h>
#include <tinynn/optim/sgd.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tinynn.h>
#include <tinynn/training/trainer.h>

int main() {
  using T = float;

  try {
    // -----------------------------
    // Model: 3 -> 4 -> ReLU -> 2
    // -----------------------------
    tinynn::Sequential<T> model;

    auto fc1 = std::make_unique<tinynn::Affine<T>>(3, 4, true);
    auto* fc1p = fc1.get();
    model.add(std::move(fc1));

    model.add(std::make_unique<tinynn::ReLU<T>>());

    auto fc2 = std::make_unique<tinynn::Affine<T>>(4, 2, true);
    auto* fc2p = fc2.get();
    model.add(std::move(fc2));

    // Deterministic initialization for reproducibility
    fc1p->init_he_normal(123);
    fc2p->init_xavier_uniform(456);

    // -----------------------------
    // Tiny toy batch
    // -----------------------------
    tinynn::Tensor<T> x(tinynn::Shape{4, 3});

    x(0, 0) = 0;
    x(0, 1) = 0;
    x(0, 2) = 1;
    x(1, 0) = 0;
    x(1, 1) = 1;
    x(1, 2) = 0;
    x(2, 0) = 1;
    x(2, 1) = 0;
    x(2, 2) = 0;
    x(3, 0) = 1;
    x(3, 1) = 1;
    x(3, 2) = 1;

    tinynn::SizeType labels_arr[4] = {0, 1, 1, 0};
    std::span<const tinynn::SizeType> labels(labels_arr, 4);

    // -----------------------------
    // Loss / Optimizer / Trainer
    // -----------------------------
    tinynn::SoftmaxWithLoss<T> loss_fn;
    tinynn::SGD<T> optimizer(/*lr=*/0.1f, /*weight_decay=*/0.0f);
    tinynn::Trainer<T> trainer(model, loss_fn, optimizer);

    // -----------------------------
    // Manual training loop with train_step()
    // -----------------------------
    std::cout << "Start trainer step-by-step MLP example..." << std::endl;

    constexpr int kSteps = 30;
    for (int step = 0; step < kSteps; ++step) {
      const auto res = trainer.train_step(x.view(), labels);

      std::cout << "train step " << step << "  loss=" << res.loss
                << "  acc=" << res.accuracy << " (" << res.num_correct << "/"
                << res.batch_size << ")\n";
    }

    // -----------------------------
    // Evaluation with eval_step()
    // -----------------------------
    const auto ev = trainer.eval_step(x.view(), labels);

    std::cout << "eval loss=" << ev.loss << "  acc=" << ev.accuracy << " ("
              << ev.num_correct << "/" << ev.batch_size << ")\n";

    std::cout << "Trainer step-by-step example finished." << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
