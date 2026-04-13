#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <tinynn/tensor/tensor.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/nn/parameter_view.h>
#include <tinynn/nn/layers/batch_norm1d.h>

namespace {

template <class T>
bool all_finite(const tinynn::Tensor<T>& t) {
  for (tinynn::SizeType i = 0; i < t.size(); ++i) {
    const T v = t[i];
    if (!std::isfinite(static_cast<double>(v))) return false;
  }
  return true;
}

template <class T>
void fill_uniform(tinynn::Tensor<T>& t, std::mt19937_64& rng, T lo, T hi) {
  std::uniform_real_distribution<double> dist(
      static_cast<double>(lo), static_cast<double>(hi));
  for (tinynn::SizeType i = 0; i < t.size(); ++i) {
    t[i] = static_cast<T>(dist(rng));
  }
}

}  // namespace

int main() {
  using T = double;

  try {
    constexpr tinynn::SizeType B = 1;
    constexpr tinynn::SizeType C = 8;

    tinynn::BatchNorm1d<T> bn(C, /*eps=*/static_cast<T>(1e-5),
                                /*momentum=*/static_cast<T>(0.1));

    std::mt19937_64 rng(123);
    tinynn::Tensor<T> x(tinynn::Shape{B, C});
    tinynn::Tensor<T> y(tinynn::Shape{B, C});
    tinynn::Tensor<T> dy(tinynn::Shape{B, C});
    tinynn::Tensor<T> dx(tinynn::Shape{B, C});

    bn.train();

    fill_uniform(x, rng, static_cast<T>(-3), static_cast<T>(3));
    fill_uniform(dy, rng, static_cast<T>(-2), static_cast<T>(2));

    bn.forward(x.view(), y.view());
    if (!all_finite(y)) {
      std::cerr << "[FAIL] train forward produced non-finite y\n";
      return 1;
    }

    bn.backward(dy.view(), dx.view());
    if (!all_finite(dx)) {
      std::cerr << "[FAIL] train backward produced non-finite dx\n";
      return 1;
    }

    std::vector<tinynn::ParameterView<T>> pvs;
    bn.collect_parameter_views(pvs);
    if (pvs.size() != 2) {
      std::cerr << "[FAIL] expected 2 ParameterViews\n";
      return 1;
    }

    // gamma/beta grads
    for (const auto& pv : pvs) {
      for (tinynn::SizeType i = 0; i < pv.grad.size(); ++i) {
        const T g = pv.grad[i];
        if (!std::isfinite(static_cast<double>(g))) {
          std::cerr << "[FAIL] train produced non-finite param grad\n";
          return 1;
        }
      }
    }

    std::cout << "[OK] B=1 train forward/backward finite\n";

    for (int k = 0; k < 8; ++k) {
      fill_uniform(x, rng, static_cast<T>(-3), static_cast<T>(3));
      bn.forward(x.view(), y.view());
    }

    bn.eval();

    fill_uniform(x, rng, static_cast<T>(-3), static_cast<T>(3));
    fill_uniform(dy, rng, static_cast<T>(-2), static_cast<T>(2));

    bn.forward(x.view(), y.view());
    if (!all_finite(y)) {
      std::cerr << "[FAIL] eval forward produced non-finite y\n";
      return 1;
    }

    bn.backward(dy.view(), dx.view());
    if (!all_finite(dx)) {
      std::cerr << "[FAIL] eval backward produced non-finite dx\n";
      return 1;
    }

    pvs.clear();
    bn.collect_parameter_views(pvs);
    for (const auto& pv : pvs) {
      for (tinynn::SizeType i = 0; i < pv.grad.size(); ++i) {
        const T g = pv.grad[i];
        if (!std::isfinite(static_cast<double>(g))) {
          std::cerr << "[FAIL] eval produced non-finite param grad\n";
          return 1;
        }
      }
    }

    std::cout << "[OK] B=1 eval forward/backward finite\n";
    std::cout << "BatchNorm1d B=1 isfinite test passed.\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 2;
  }
}
