
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include <tinynn/tensor/tensor.h>
#include <tinynn/tensor/tensor_view.h>
#include <tinynn/nn/parameter_view.h>
#include <tinynn/nn/layers/batch_norm1d.h>  // <- あなたが作ったBN

namespace {

template <class T>
T abs_val(T x) {
  return x < static_cast<T>(0) ? -x : x;
}

template <class T>
bool near(T a, T b, T atol, T rtol) {
  const T diff = abs_val(a - b);
  const T tol = atol + rtol * abs_val(b);
  return diff <= tol;
}

template <class T>
void fill_uniform(tinynn::Tensor<T>& t, std::mt19937_64& rng, T lo, T hi) {
  std::uniform_real_distribution<double> dist(
      static_cast<double>(lo), static_cast<double>(hi));
  for (tinynn::SizeType i = 0; i < t.size(); ++i) {
    t[i] = static_cast<T>(dist(rng));
  }
}

template <class T>
T dot_loss(const tinynn::Tensor<T>& y, const tinynn::Tensor<T>& upstream) {
  if (y.size() != upstream.size()) {
    throw std::invalid_argument("dot_loss: size mismatch");
  }
  T s = static_cast<T>(0);
  for (tinynn::SizeType i = 0; i < y.size(); ++i) {
    s += y[i] * upstream[i];
  }
  return s;
}

// Loss(x, gamma, beta) = sum_{b,c} y[b,c] * upstream[b,c]
template <class T>
T compute_loss_train(
    tinynn::BatchNorm1d<T>& bn,
    const tinynn::Tensor<T>& x,
    const tinynn::Tensor<T>& upstream) {

  tinynn::Tensor<T> y(x.shape());
  bn.forward(x.view(), y.view());
  return dot_loss(y, upstream);
}

// Central difference gradient for a Tensor (owned) w.r.t its i-th element.
template <class T, class LossFn>
T numeric_grad_elem_owned(tinynn::Tensor<T>& param, tinynn::SizeType i, T h, LossFn&& loss_fn) {
  const T old = param[i];

  param[i] = old + h;
  const T f1 = loss_fn();

  param[i] = old - h;
  const T f2 = loss_fn();

  param[i] = old;
  return (f1 - f2) / (static_cast<T>(2) * h);
}

// Central difference gradient for a TensorView (non-owning) w.r.t its i-th element.
template <class T, class LossFn>
T numeric_grad_elem_view(tinynn::TensorView<T> param, tinynn::SizeType i, T h, LossFn&& loss_fn) {
  T* p = param.data();
  const T old = p[i];

  p[i] = old + h;
  const T f1 = loss_fn();

  p[i] = old - h;
  const T f2 = loss_fn();

  p[i] = old;
  return (f1 - f2) / (static_cast<T>(2) * h);
}

}  // namespace

int main() {
  using T = double;

  try {
    constexpr tinynn::SizeType B = 4;
    constexpr tinynn::SizeType C = 3;

    tinynn::BatchNorm1d<T> bn(/*num_features=*/C,
                             /*eps=*/static_cast<T>(1e-5),
                             /*momentum=*/static_cast<T>(0));
    bn.train();

    std::mt19937_64 rng(12345);

    tinynn::Tensor<T> x({B, C});
    tinynn::Tensor<T> upstream({B, C});  // dL/dy
    fill_uniform(x, rng, static_cast<T>(-1.0), static_cast<T>(1.0));
    fill_uniform(upstream, rng, static_cast<T>(-1.0), static_cast<T>(1.0));

    // --- Analytic grads ---
    // forward to set cache
    tinynn::Tensor<T> y({B, C});
    bn.forward(x.view(), y.view());

    tinynn::Tensor<T> dx({B, C});
    bn.backward(upstream.view(), dx.view());

    // collect gamma/beta + their grads
    std::vector<tinynn::ParameterView<T>> pvs;
    bn.collect_parameter_views(pvs);
    if (pvs.size() != 2) {
      throw std::runtime_error("expected 2 parameter views (gamma, beta)");
    }

    tinynn::TensorView<T> gamma = pvs[0].param;
    tinynn::TensorView<const T> dgamma = pvs[0].grad;

    tinynn::TensorView<T> beta = pvs[1].param;
    tinynn::TensorView<const T> dbeta = pvs[1].grad;

    // --- Numeric grads ---
    const T h = static_cast<T>(1e-6);     // central difference step
    const T atol = static_cast<T>(1e-5);  // abs tol
    const T rtol = static_cast<T>(1e-4);  // rel tol

    auto loss_fn = [&]() -> T {
      return compute_loss_train(bn, x, upstream);
    };

    // 1) dx check
    {
      T max_abs = 0;
      tinynn::SizeType worst_i = 0;

      for (tinynn::SizeType i = 0; i < x.size(); ++i) {
        const T g_num = numeric_grad_elem_owned(x, i, h, loss_fn);
        const T g_ana = dx[i];
        const T diff = abs_val(g_num - g_ana);
        if (diff > max_abs) {
          max_abs = diff;
          worst_i = i;
        }
        if (!near(g_num, g_ana, atol, rtol)) {
          std::cerr << "[FAIL] dx i=" << i
                    << " num=" << g_num
                    << " ana=" << g_ana
                    << " diff=" << diff << "\n";
          return 1;
        }
      }
      std::cout << "[OK] dx numeric check passed. max_abs_diff=" << max_abs
                << " at i=" << worst_i << "\n";
    }

    // 2) dgamma check
    {
      T max_abs = 0;
      tinynn::SizeType worst_i = 0;

      for (tinynn::SizeType i = 0; i < gamma.size(); ++i) {
        const T g_num = numeric_grad_elem_view(gamma, i, h, loss_fn);
        const T g_ana = dgamma[i];
        const T diff = abs_val(g_num - g_ana);
        if (diff > max_abs) {
          max_abs = diff;
          worst_i = i;
        }
        if (!near(g_num, g_ana, atol, rtol)) {
          std::cerr << "[FAIL] dgamma i=" << i
                    << " num=" << g_num
                    << " ana=" << g_ana
                    << " diff=" << diff << "\n";
          return 1;
        }
      }
      std::cout << "[OK] dgamma numeric check passed. max_abs_diff=" << max_abs
                << " at i=" << worst_i << "\n";
    }

    // 3) dbeta check
    {
      T max_abs = 0;
      tinynn::SizeType worst_i = 0;

      for (tinynn::SizeType i = 0; i < beta.size(); ++i) {
        const T g_num = numeric_grad_elem_view(beta, i, h, loss_fn);
        const T g_ana = dbeta[i];
        const T diff = abs_val(g_num - g_ana);
        if (diff > max_abs) {
          max_abs = diff;
          worst_i = i;
        }
        if (!near(g_num, g_ana, atol, rtol)) {
          std::cerr << "[FAIL] dbeta i=" << i
                    << " num=" << g_num
                    << " ana=" << g_ana
                    << " diff=" << diff << "\n";
          return 1;
        }
      }
      std::cout << "[OK] dbeta numeric check passed. max_abs_diff=" << max_abs
                << " at i=" << worst_i << "\n";
    }

    std::cout << "All numeric gradient checks passed.\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 2;
  }
}
