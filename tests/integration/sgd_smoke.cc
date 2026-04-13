#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include <tinynn/nn/sequential.h>
#include <tinynn/nn/parameter_view.h>
#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/optim/sgd.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/tinynn.h>

namespace {

template <class T>
bool is_finite(T x) {
  return std::isfinite(static_cast<double>(x));
}

template <class T>
void sgd_step(std::vector<tinynn::ParameterView<T>>& params, T lr) {
  for (auto& pv : params) {
    auto p = pv.param;  // TensorView<T>
    auto g = pv.grad;   // TensorView<const T>
    assert(p.size() == g.size());
    for (tinynn::SizeType i = 0; i < p.size(); ++i) {
      p[i] -= lr * g[i];
    }
  }
}

}  // namespace

int main() {
  using T = float;

  // -------------------------
  // Model: Affine(3 -> 2) (logits)
  // -------------------------
  tinynn::Sequential<T> model;
  auto fc = std::make_unique<tinynn::Affine<T>>(/*in_dim=*/3, /*out_dim=*/2, /*with_bias=*/true);
  auto* fcp = fc.get();
  model.add(std::move(fc));

  // -------------------------
  // Initialize parameters to zeros (deterministic)
  // -------------------------
  {
    auto W = fcp->W();  // shape [3,2]
    auto b = fcp->b();  // shape [2]
    for (tinynn::SizeType i = 0; i < W.size(); ++i) W[i] = 0.0f;
    for (tinynn::SizeType i = 0; i < b.size(); ++i) b[i] = 0.0f;
  }

  // -------------------------
  // Tiny batch (2 samples), 2 classes
  // x0 -> class 0, x1 -> class 1
  // -------------------------
  tinynn::Tensor<T> x(tinynn::Shape{2, 3});
  // sample 0: [1,0,0]
  x(0, 0) = 1.0f; x(0, 1) = 0.0f; x(0, 2) = 0.0f;
  // sample 1: [0,1,0]
  x(1, 0) = 0.0f; x(1, 1) = 1.0f; x(1, 2) = 0.0f;

  const tinynn::SizeType labels_arr[2] = {0, 1};
  std::span<const tinynn::SizeType> labels(labels_arr, 2);

  tinynn::SoftmaxWithLoss<T> loss_fn;

  // -------------------------
  // Forward -> Loss (before)
  // -------------------------
  auto logits0 = model.forward(x.view());   // [2,2]
  T loss0 = loss_fn.forward(logits0, labels);
  std::cout << "loss0=" << loss0 << "\n";
  assert(is_finite(loss0));

  // -------------------------
  // Backward
  // -------------------------
  auto dlogits = loss_fn.backward();  // (p - t)/batch
  model.backward(dlogits);

  // -------------------------
  // SGD step
  // -------------------------
  tinynn::SGD<float> opt(/*lr=*/0.5f, /*weight_decay=*/0.0f);

  std::vector<tinynn::ParameterView<float>> params;
  model.collect_parameter_views(params);
  opt.step(std::span<tinynn::ParameterView<float>>(params));

  // -------------------------
  // Forward -> Loss (after one step)
  // -------------------------
  auto logits1 = model.forward(x.view());
  T loss1 = loss_fn.forward(logits1, labels);
  std::cout << "loss1=" << loss1 << "\n";
  assert(is_finite(loss1));

  // Expect loss to go down (should, for this setup).
  // Use a small tolerance to avoid edge-case numerical noise.
  assert(loss1 < loss0 + 1e-6f);

  std::cout << "1-step SGD smoke test OK.\n";
  return 0;
}
