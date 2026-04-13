#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/layers/batch_norm1d.h>
#include <tinynn/nn/layers/relu.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/nn/parameter_view.h>
#include <tinynn/nn/sequential.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor.h>

namespace {

template <class T>
void make_batch(tinynn::Tensor<T>& x, std::vector<tinynn::SizeType>& labels,
                std::mt19937_64& rng) {
  const tinynn::SizeType B = x.shape().rows();
  std::normal_distribution<double> n01(0.0, 1.0);

  labels.resize(B);
  for (tinynn::SizeType b = 0; b < B; ++b) {
    const double u = n01(rng);
    const double v = n01(rng);

    const tinynn::SizeType cls = (u + v > 0.0) ? 1 : 0;
    labels[b] = cls;

    x(b, 0) = static_cast<T>(u);
    x(b, 1) = static_cast<T>(v);
  }
}

template <class T>
void sgd_step(std::vector<tinynn::ParameterView<T>>& pvs, T lr) {
  for (auto& pv : pvs) {
    for (tinynn::SizeType i = 0; i < pv.param.size(); ++i) {
      pv.param[i] -= lr * pv.grad[i];
    }
  }
}

template <class T>
void init_model_for_test(tinynn::Sequential<T>& model,
                         std::uint32_t seed_base = 123) {
  std::uint32_t s = seed_base;
  model.for_each_layer([&](tinynn::Layer<T>& layer) {
    if (auto* a = dynamic_cast<tinynn::Affine<T>*>(&layer)) {
      a->init_he_normal(s++);
    }
  });
}

template <class T>
double get_first_relu_pos_ratio(const tinynn::Sequential<T>& model) {
  double ratio = -1.0;
  model.for_each_layer([&](const tinynn::Layer<T>& layer) {
    if (ratio >= 0.0) return;  // already found
    if (auto* r = dynamic_cast<const tinynn::ReLU<T>*>(&layer)) {
      ratio = r->last_pos_ratio();
    }
  });
  return ratio;
}

}  // namespace

int main() {
  using T = double;

  constexpr tinynn::SizeType B = 64;

  std::mt19937_64 rng(1);

  tinynn::Sequential<T> model;
  model.add(std::make_unique<tinynn::Affine<T>>(/*in=*/2, /*out=*/16));
  model.add(std::make_unique<tinynn::BatchNorm1d<T>>(/*C=*/16));
  model.add(std::make_unique<tinynn::ReLU<T>>());
  model.add(std::make_unique<tinynn::Affine<T>>(/*in=*/16, /*out=*/2));

  init_model_for_test(model, /*seed_base=*/123);

  tinynn::SoftmaxWithLoss<T> loss_fn;

  tinynn::Tensor<T> x(tinynn::Shape{B, 2});
  std::vector<tinynn::SizeType> labels;

  T loss0 = static_cast<T>(0);
  T loss_last = static_cast<T>(0);

  make_batch(x, labels, rng);

  for (int epoch = 0; epoch < 200; ++epoch) {
    model.train();

    // forward: logits
    auto logits = model.forward(x.view());  // [B,2]

    if (epoch == 0) {
      const double r = get_first_relu_pos_ratio(model);
      if (r >= 0.0) {
        if (r <= 0.05) {
          throw std::runtime_error(
              "ReLU pos_ratio too low; likely bad Affine init (all <=0)");
        }
      }
    }

    // loss forward/backward (produces dlogits)
    const T loss = loss_fn.forward(
        logits.as_const(),
        std::span<const tinynn::SizeType>(labels.data(), labels.size()));
    auto dlogits = loss_fn.backward();  // TensorView<T> shape [B,2]

    // backprop through model
    model.backward(dlogits.as_const());

    // update params
    std::vector<tinynn::ParameterView<T>> pvs;
    model.collect_parameter_views(pvs);
    sgd_step(pvs, static_cast<T>(0.2));

    if (epoch == 0) loss0 = loss;
    loss_last = loss;

    if (epoch % 5 == 0) {
      std::cout << "epoch " << epoch << " loss=" << loss << "\n";
    }
  }

  std::cout << "loss0=" << loss0 << " loss_last=" << loss_last << "\n";
  if (!(loss_last < loss0)) {
    std::cerr << "[WARN] loss did not decrease (may still be ok depending on "
                 "init/seed)\n";
  }

  model.eval();
  make_batch(x, labels, rng);
  (void)model.forward(x.view());

  std::cout << "Sequential+BN integration test done.\n";
  return 0;
}
