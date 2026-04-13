// optimizer_weight_decay_filter_test.cc
//
// Verifies weight decay filter policy:
// - Apply weight_decay to non-bias, non-1D parameters (e.g., 2D weight matrices)
// - Do NOT apply weight_decay to bias parameters
// - Do NOT apply weight_decay to 1D parameters (e.g., BN gamma/beta)
//
// This test is intentionally minimal and does not depend on Affine/BN layers.
// It uses ParameterView directly with tiny tensors.
//
// Expected policy (as discussed):
//   should_apply_weight_decay(pv) == (pv.kind != kBias) && (pv.param.rank != 1)
//
// Build: add to your test target and compile with C++20.

#include <iostream>
#include <cmath>
#include <stdexcept>

#include <tinynn/nn/parameter_view.h>
#include <tinynn/optim/weight_decay_filter.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tinynn.h>

namespace {

template <class T>
void require_close(T a, T b, T tol, const char* msg) {
  const T diff = static_cast<T>(std::fabs(static_cast<double>(a - b)));
  if (diff > tol) {
    std::cerr << "[NG] " << msg << ": a=" << a << " b=" << b << " diff=" << diff
              << " tol=" << tol << "\n";
    throw std::runtime_error(msg);
  }
}

template <class T>
void apply_l2_wd_step_like_sgd(tinynn::ParameterView<T>& pv, T lr, T wd) {
  auto p = pv.param;
  auto g = pv.grad;

  if (p.size() != g.size()) {
    throw std::invalid_argument("apply_l2_wd_step_like_sgd: size mismatch");
  }

  const bool apply_wd = (wd != static_cast<T>(0)) && tinynn::should_apply_weight_decay(pv);

  for (tinynn::SizeType i = 0; i < p.size(); ++i) {
    T grad = g[i];
    if (apply_wd) {
      grad += wd * p[i];
    }
    p[i] -= lr * grad;
  }
}

template <class T>
void apply_decoupled_wd_only(tinynn::ParameterView<T>& pv, T lr, T wd) {
  auto p = pv.param;
  if (wd == static_cast<T>(0)) return;

  const bool apply_wd = tinynn::should_apply_weight_decay(pv);
  if (!apply_wd) return;

  for (tinynn::SizeType i = 0; i < p.size(); ++i) {
    p[i] -= lr * wd * p[i];
  }
}

}  // namespace

int main() {
  using T = double;

  const T lr = 0.1;
  const T wd = 0.5;
  const T tol = 1e-12;

  // ---- Case 1: 2D weight matrix => WD should apply ----
  {
    tinynn::Tensor<T> W(tinynn::Shape{2, 3});
    tinynn::Tensor<T> dW(tinynn::Shape{2, 3});

    // init param and grad
    for (tinynn::SizeType i = 0; i < W.size(); ++i) W[i] = static_cast<T>(1.0);
    for (tinynn::SizeType i = 0; i < dW.size(); ++i) dW[i] = static_cast<T>(0.0);

    tinynn::ParameterView<T> pv{
        /*param=*/W.view(),
        /*grad=*/dW.view(),
        /*kind=*/tinynn::ParamKind::kWeight,
        /*id=*/1,
    };

    if (!tinynn::should_apply_weight_decay(pv)) {
      throw std::runtime_error("Case1: expected should_apply_weight_decay=true");
    }

    // L2-style: p -= lr*(wd*p)
    apply_l2_wd_step_like_sgd(pv, lr, wd);

    // expect each element: 1 - 0.1*(0.5*1) = 0.95
    for (tinynn::SizeType i = 0; i < W.size(); ++i) {
      require_close(W[i], static_cast<T>(0.95), tol, "Case1 L2 wd update mismatch");
    }

    // Decoupled-style: p -= lr*wd*p (starting from 1 again)
    for (tinynn::SizeType i = 0; i < W.size(); ++i) W[i] = static_cast<T>(1.0);
    apply_decoupled_wd_only(pv, lr, wd);

    for (tinynn::SizeType i = 0; i < W.size(); ++i) {
      require_close(W[i], static_cast<T>(0.95), tol, "Case1 decoupled wd update mismatch");
    }

    std::cout << "[OK] Case1 (2D weight) weight_decay applied\n";
  }

  // ---- Case 2: bias (1D) => WD should NOT apply ----
  {
    tinynn::Tensor<T> b(tinynn::Shape{3});
    tinynn::Tensor<T> db(tinynn::Shape{3});

    for (tinynn::SizeType i = 0; i < b.size(); ++i) b[i] = static_cast<T>(1.0);
    for (tinynn::SizeType i = 0; i < db.size(); ++i) db[i] = static_cast<T>(0.0);

    tinynn::ParameterView<T> pv{
        /*param=*/b.view(),
        /*grad=*/db.view(),
        /*kind=*/tinynn::ParamKind::kBias,
        /*id=*/2,
    };

    if (tinynn::should_apply_weight_decay(pv)) {
      throw std::runtime_error("Case2: expected should_apply_weight_decay=false (bias)");
    }

    apply_l2_wd_step_like_sgd(pv, lr, wd);
    for (tinynn::SizeType i = 0; i < b.size(); ++i) {
      require_close(b[i], static_cast<T>(1.0), tol, "Case2 bias should not decay (L2)");
    }

    apply_decoupled_wd_only(pv, lr, wd);
    for (tinynn::SizeType i = 0; i < b.size(); ++i) {
      require_close(b[i], static_cast<T>(1.0), tol, "Case2 bias should not decay (decoupled)");
    }

    std::cout << "[OK] Case2 (bias) weight_decay NOT applied\n";
  }

  // ---- Case 3: 1D non-bias (e.g., BN gamma) => WD should NOT apply ----
  {
    tinynn::Tensor<T> gamma(tinynn::Shape{3});
    tinynn::Tensor<T> dgamma(tinynn::Shape{3});

    for (tinynn::SizeType i = 0; i < gamma.size(); ++i) gamma[i] = static_cast<T>(1.0);
    for (tinynn::SizeType i = 0; i < dgamma.size(); ++i) dgamma[i] = static_cast<T>(0.0);

    tinynn::ParameterView<T> pv{
        /*param=*/gamma.view(),
        /*grad=*/dgamma.view(),
        /*kind=*/tinynn::ParamKind::kWeight,  // important: still "weight"
        /*id=*/3,
    };

    if (tinynn::should_apply_weight_decay(pv)) {
      throw std::runtime_error("Case3: expected should_apply_weight_decay=false (1D weight)");
    }

    apply_l2_wd_step_like_sgd(pv, lr, wd);
    for (tinynn::SizeType i = 0; i < gamma.size(); ++i) {
      require_close(gamma[i], static_cast<T>(1.0), tol, "Case3 1D weight should not decay (L2)");
    }

    apply_decoupled_wd_only(pv, lr, wd);
    for (tinynn::SizeType i = 0; i < gamma.size(); ++i) {
      require_close(gamma[i], static_cast<T>(1.0), tol, "Case3 1D weight should not decay (decoupled)");
    }

    std::cout << "[OK] Case3 (1D weight like BN gamma) weight_decay NOT applied\n";
  }

  std::cout << "optimizer_weight_decay_filter_test passed.\n";
  return 0;
}
