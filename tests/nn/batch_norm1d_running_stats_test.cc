#include <iostream>
#include <random>

#include <tinynn/tensor/tensor.h>
#include <tinynn/nn/layers/batch_norm1d.h>

using T = double;

int main() {
  constexpr tinynn::SizeType B = 8;
  constexpr tinynn::SizeType C = 4;

  tinynn::BatchNorm1d<T> bn(C, 1e-5, 0.1);  // momentum>0
  bn.train();

  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  tinynn::Tensor<T> x(tinynn::Shape{B, C});
  tinynn::Tensor<T> y(tinynn::Shape{B, C});

  auto mean_before = bn.running_mean();
  auto var_before  = bn.running_var();

  for (int iter = 0; iter < 5; ++iter) {
    for (tinynn::SizeType i = 0; i < x.size(); ++i)
      x[i] = static_cast<T>(dist(rng));

    bn.forward(x.view(), y.view());
  }

  auto mean_after_train = bn.running_mean();
  auto var_after_train  = bn.running_var();

  bool changed = false;
  for (tinynn::SizeType i = 0; i < C; ++i) {
    if (mean_before[i] != mean_after_train[i] ||
        var_before[i]  != var_after_train[i]) {
      changed = true;
      break;
    }
  }

  if (!changed) {
    std::cerr << "[FAIL] running stats did not change during training\n";
    return 1;
  }

  std::cout << "[OK] running stats updated during train\n";

  // ---- Eval forward ----
  bn.eval();

  auto mean_before_eval = bn.running_mean();
  auto var_before_eval  = bn.running_var();

  for (int iter = 0; iter < 5; ++iter) {
    for (tinynn::SizeType i = 0; i < x.size(); ++i)
      x[i] = static_cast<T>(dist(rng));

    bn.forward(x.view(), y.view());
  }

  auto mean_after_eval = bn.running_mean();
  auto var_after_eval  = bn.running_var();

  bool changed_eval = false;
  for (tinynn::SizeType i = 0; i < C; ++i) {
    if (mean_before_eval[i] != mean_after_eval[i] ||
        var_before_eval[i]  != var_after_eval[i]) {
      changed_eval = true;
      break;
    }
  }

  if (changed_eval) {
    std::cerr << "[FAIL] running stats changed during eval\n";
    return 1;
  }

  std::cout << "[OK] running stats frozen during eval\n";
  std::cout << "BatchNorm1d running stats test passed.\n";

  return 0;
}
