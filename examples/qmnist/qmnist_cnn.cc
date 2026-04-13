#include <iostream>
#include <memory>

#include <tinynn/data/data_loader.h>
#include <tinynn/datasets/qmnist_image_dataset.h>
#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/layers/batch_norm2d.h>
#include <tinynn/nn/layers/conv2d.h>
#include <tinynn/nn/layers/dropout.h>
#include <tinynn/nn/layers/dropout2d.h>
#include <tinynn/nn/layers/flatten.h>
#include <tinynn/nn/layers/maxpool2d.h>
#include <tinynn/nn/layers/relu.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/nn/sequential.h>
#include <tinynn/optim/adam.h>
#include <tinynn/training/callbacks/csv_logger.h>
#include <tinynn/training/callbacks/epoch_summary.h>
#include <tinynn/training/callbacks/print_every.h>
#include <tinynn/training/trainer.h>

int main() {
  using T = float;

  try {
    // =============================
    // Dataset / DataLoader
    // =============================
    tinynn::QmnistImageDataset<T>::Files train_files{
        "data/qmnist/qmnist-train-images-idx3-ubyte",
        "data/qmnist/qmnist-train-labels-idx2-int"};

    tinynn::QmnistImageDataset<T>::Files test_files{
        "data/qmnist/qmnist-test-images-idx3-ubyte",
        "data/qmnist/qmnist-test-labels-idx2-int"};

    tinynn::QmnistImageDataset<T> train_ds(train_files);
    tinynn::QmnistImageDataset<T> test_ds(test_files);

    tinynn::DataLoader<T, tinynn::QmnistImageDataset<T>> train_loader(
        train_ds, /*batch_size=*/64, /*shuffle=*/true, /*seed=*/123);

    tinynn::DataLoader<T, tinynn::QmnistImageDataset<T>> test_loader(
        test_ds, /*batch_size=*/64, /*shuffle=*/false, /*seed=*/123);

    // =============================
    // Model (CNN)
    // Input: [N, 1, 28, 28]
    // =============================
    tinynn::Sequential<T> model;

    // ---- Block 1 ----
    model.add(std::make_unique<tinynn::Conv2d<T>>(tinynn::Conv2dOptions{
        1, 16,  // in/out channels
        3, 3,   // kernel
        true,   // bias
        1, 1,   // stride
        1, 1    // padding
    }));
    model.add(std::make_unique<tinynn::BatchNorm2d<T>>(16));
    model.add(std::make_unique<tinynn::ReLU<T>>());
    model.add(std::make_unique<tinynn::MaxPool2d<T>>(2, 2));

    // ---- Block 2 ----
    model.add(std::make_unique<tinynn::Conv2d<T>>(
        tinynn::Conv2dOptions{16, 32, 3, 3, true, 1, 1, 1, 1}));
    model.add(std::make_unique<tinynn::BatchNorm2d<T>>(32));
    model.add(std::make_unique<tinynn::ReLU<T>>());
    model.add(std::make_unique<tinynn::MaxPool2d<T>>(2, 2));

    model.add(std::make_unique<tinynn::Dropout2d<T>>(static_cast<T>(0.25)));

    // ---- Classifier ----
    model.add(std::make_unique<tinynn::Flatten<T>>());
    model.add(std::make_unique<tinynn::Affine<T>>(32 * 7 * 7, 128, true));
    model.add(std::make_unique<tinynn::ReLU<T>>());
    model.add(std::make_unique<tinynn::Dropout<T>>(static_cast<T>(0.5)));
    model.add(std::make_unique<tinynn::Affine<T>>(128, 10, true));

    // =============================
    // Loss / Optimizer / Trainer
    // =============================
    tinynn::SoftmaxWithLoss<T> loss_fn;
    tinynn::Adam<T> optimizer(static_cast<T>(0.001));

    tinynn::Trainer<T> trainer(model, loss_fn, optimizer);

    // =============================
    // Callbacks
    // =============================
    tinynn::PrintEveryCallback<T> print_cb(100);
    tinynn::EpochSummaryCallback<T> epoch_cb;
    tinynn::CSVLoggerCallback<T> csv_cb("qmnist_cnn_log.csv");

    trainer.add_callback(&print_cb);
    trainer.add_callback(&epoch_cb);
    trainer.add_callback(&csv_cb);

    // =============================
    // Train
    // =============================
    tinynn::FitOptions opt;
    opt.epochs = 5;

    std::cout << "Start training..." << std::endl;
    trainer.fit(train_loader, test_loader, opt);
    std::cout << "Training finished." << std::endl;

    // =============================
    // Evaluate
    // =============================
    const auto ev = trainer.eval_epoch(test_loader, opt.epochs - 1);

    std::cout << "Final eval loss=" << ev.loss << " acc=" << ev.accuracy << " ("
              << ev.num_correct << "/" << ev.num_samples << ")" << std::endl;

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
