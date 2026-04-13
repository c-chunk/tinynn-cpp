#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <tinynn/nn/sequential.h>
#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/layers/relu.h>
#include <tinynn/nn/losses/softmax_with_loss.h>
#include <tinynn/optim/sgd.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tinynn.h>

int main() {
  using T = float;

  // -----------------------
  // Model: 3 -> 4 -> ReLU -> 2
  // -----------------------
  tinynn::Sequential<T> model;

#if 0
  model.add(std::make_unique<tinynn::Affine<T>>(3, 4, true));
  model.add(std::make_unique<tinynn::ReLU<T>>());
  model.add(std::make_unique<tinynn::Affine<T>>(4, 2, true));
#else

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
#endif

  // -----------------------
  // Tiny dataset (4 samples)
  // -----------------------
  tinynn::Tensor<T> x(tinynn::Shape{4, 3});

  x(0,0)=0; x(0,1)=0; x(0,2)=1;
  x(1,0)=0; x(1,1)=1; x(1,2)=0;
  x(2,0)=1; x(2,1)=0; x(2,2)=0;
  x(3,0)=1; x(3,1)=1; x(3,2)=1;

  tinynn::SizeType labels_arr[4] = {0,1,1,0};
  std::span<const tinynn::SizeType> labels(labels_arr,4);

  tinynn::SoftmaxWithLoss<T> loss_fn;
  tinynn::SGD<T> opt(0.1f);

  // -----------------------
  // Training loop (20 steps)
  // -----------------------
  T prev_loss = std::numeric_limits<T>::max();

  for(int step=0; step<20; ++step) {

    auto logits = model.forward(x.view());
    T loss = loss_fn.forward(logits, labels);

    auto dlogits = loss_fn.backward();
    model.backward(dlogits);

    std::vector<tinynn::ParameterView<T>> params;
    model.collect_parameter_views(params);
    opt.step(params);

    std::cout << "W0=" << params[0].param[0] << "\n";
    std::cout << "g0=" << params[0].grad[0] << "\n";
    std::cout << "step " << step << " loss=" << loss << "\n\n";

    assert(std::isfinite(loss));
    if(step > 0) {
      // loss should not explode
      assert(loss < prev_loss + 1.0f);
    }
    prev_loss = loss;
  }

  std::cout << "2-layer MLP smoke test OK.\n";
  return 0;
}
