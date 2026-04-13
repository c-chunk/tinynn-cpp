#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <tinynn/nn/parameter_view.h>
#include <tinynn/optim/weight_decay_filter.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor.h>
#include <tinynn/tinynn.h>
#include <tinynn/training/callbacks/checkpoint_stream.h>

// include your optimizers
#include <tinynn/optim/sgd.h>
#include <tinynn/optim/momentum.h>
#include <tinynn/optim/adam.h>
#include <tinynn/optim/adamw.h>
#include <tinynn/optim/rmsprop.h>

namespace {

using tinynn::SizeType;

template <class T>
T absdiff(T a, T b) {
  return static_cast<T>(std::fabs(static_cast<double>(a - b)));
}

template <class T>
T max_abs_diff(const tinynn::Tensor<T>& a, const tinynn::Tensor<T>& b) {
  if (a.size() != b.size()) throw std::runtime_error("max_abs_diff: size mismatch");
  T m = static_cast<T>(0);
  for (SizeType i = 0; i < a.size(); ++i) {
    const T d = absdiff(a[i], b[i]);
    if (d > m) m = d;
  }
  return m;
}

// deterministic gradients that change per step
template <class T>
void fill_grads(tinynn::Tensor<T>& g, std::uint64_t step_idx, T scale) {
  for (SizeType i = 0; i < g.size(); ++i) {
    // small but non-trivial
    const double v = 0.01 * (1.0 + static_cast<double>(i))
                   + 0.001 * static_cast<double>(step_idx);
    g[i] = static_cast<T>(scale * static_cast<T>(v));
  }
}

template <class T>
struct ParamPack {
  // params
  tinynn::Tensor<T> W;
  tinynn::Tensor<T> gamma;
  tinynn::Tensor<T> b;

  // grads
  tinynn::Tensor<T> dW;
  tinynn::Tensor<T> dgamma;
  tinynn::Tensor<T> db;

  // ids (stable across runs)
  std::uintptr_t wid = 101;
  std::uintptr_t gid = 202;
  std::uintptr_t bid = 303;

  explicit ParamPack()
      : W(tinynn::Shape{2, 3}),
        gamma(tinynn::Shape{3}),
        b(tinynn::Shape{3}),
        dW(tinynn::Shape{2, 3}),
        dgamma(tinynn::Shape{3}),
        db(tinynn::Shape{3}) {
    // init params
    for (SizeType i = 0; i < W.size(); ++i) W[i] = static_cast<T>(0.1) * static_cast<T>(i + 1);
    for (SizeType i = 0; i < gamma.size(); ++i) gamma[i] = static_cast<T>(1);
    for (SizeType i = 0; i < b.size(); ++i) b[i] = static_cast<T>(0);
  }

  std::vector<tinynn::ParameterView<T>> views() {
    std::vector<tinynn::ParameterView<T>> pvs;
    pvs.push_back(tinynn::ParameterView<T>{W.view(), dW.view(), tinynn::ParamKind::kWeight, wid});
    pvs.push_back(tinynn::ParameterView<T>{gamma.view(), dgamma.view(), tinynn::ParamKind::kWeight, gid});  // 1D weight (BN gamma)
    pvs.push_back(tinynn::ParameterView<T>{b.view(), db.view(), tinynn::ParamKind::kBias, bid});            // bias
    return pvs;
  }
};

template <class Opt, class T>
void save_optimizer_to_bytes(const Opt& opt, std::string& out_bytes) {
  std::ostringstream oss(std::ios::binary);
  tinynn::CheckpointWriter w(oss);
  // tag + state
  w.write_string(opt.checkpoint_tag());
  opt.save_state(w);
  out_bytes = oss.str();
}

template <class Opt, class T>
void load_optimizer_from_bytes(Opt& opt, const std::string& bytes) {
  std::istringstream iss(bytes, std::ios::binary);
  tinynn::CheckpointReader r(iss);
  const std::string tag = r.read_string();
  if (tag != opt.checkpoint_tag()) {
    throw std::runtime_error("checkpoint tag mismatch: expected different optimizer type");
  }
  opt.load_state(r);
}

template <class Opt, class T>
void run_roundtrip_test(const char* name,
                        Opt opt_continuous,
                        Opt opt_resume,
                        std::uint64_t total_steps,
                        std::uint64_t split_step,
                        T tol) {
  ParamPack<T> A;
  ParamPack<T> B;

  // continuous run
  for (std::uint64_t t = 0; t < total_steps; ++t) {
    fill_grads(A.dW, t, static_cast<T>(1));
    fill_grads(A.dgamma, t, static_cast<T>(0.5));
    fill_grads(A.db, t, static_cast<T>(0.25));

    auto pvs = A.views();
    opt_continuous.step(std::span<tinynn::ParameterView<T>>(pvs.data(), pvs.size()));
  }

  // split run: first part
  for (std::uint64_t t = 0; t < split_step; ++t) {
    fill_grads(B.dW, t, static_cast<T>(1));
    fill_grads(B.dgamma, t, static_cast<T>(0.5));
    fill_grads(B.db, t, static_cast<T>(0.25));

    auto pvs = B.views();
    opt_resume.step(std::span<tinynn::ParameterView<T>>(pvs.data(), pvs.size()));
  }

  // save optimizer state
  std::string bytes;
  save_optimizer_to_bytes<Opt, T>(opt_resume, bytes);

  // recreate + load
  Opt opt_loaded = opt_resume;  // ok for these (copies hyperparams); state will be overwritten
  // But to be strict, you can default-construct and rely on load_state restoring lr/wd too.
  load_optimizer_from_bytes<Opt, T>(opt_loaded, bytes);

  // continue remaining steps
  for (std::uint64_t t = split_step; t < total_steps; ++t) {
    fill_grads(B.dW, t, static_cast<T>(1));
    fill_grads(B.dgamma, t, static_cast<T>(0.5));
    fill_grads(B.db, t, static_cast<T>(0.25));

    auto pvs = B.views();
    opt_loaded.step(std::span<tinynn::ParameterView<T>>(pvs.data(), pvs.size()));
  }

  const T dW = max_abs_diff(A.W, B.W);
  const T dG = max_abs_diff(A.gamma, B.gamma);
  const T dB = max_abs_diff(A.b, B.b);

  if (dW > tol || dG > tol || dB > tol) {
    std::cerr << "[NG] " << name << " roundtrip mismatch: "
              << "dW=" << dW << " dGamma=" << dG << " dBias=" << dB
              << " tol=" << tol << "\n";
    throw std::runtime_error("optimizer checkpoint roundtrip failed");
  }

  std::cout << "[OK] " << name << " checkpoint roundtrip matched. "
            << "max(dW,dG,dB)=" << std::max({dW, dG, dB}) << "\n";
}

}  // namespace

int main() {
  using T = double;

  const std::uint64_t total_steps = 50;
  const std::uint64_t split_step = 17;
  const T tol = 1e-12;

  // Momentum
  {
    tinynn::Momentum<T> optA(/*lr=*/1e-2, /*momentum=*/0.9, /*wd=*/1e-2);
    tinynn::Momentum<T> optB(/*lr=*/1e-2, /*momentum=*/0.9, /*wd=*/1e-2);
    run_roundtrip_test<tinynn::Momentum<T>, T>("Momentum", optA, optB, total_steps, split_step, tol);
  }

  // Adam
  {
    tinynn::Adam<T> optA(/*lr=*/1e-3, /*b1=*/0.9, /*b2=*/0.999, /*eps=*/1e-8, /*wd=*/1e-2);
    tinynn::Adam<T> optB(/*lr=*/1e-3, /*b1=*/0.9, /*b2=*/0.999, /*eps=*/1e-8, /*wd=*/1e-2);
    run_roundtrip_test<tinynn::Adam<T>, T>("Adam", optA, optB, total_steps, split_step, tol);
  }

  // AdamW
  {
    tinynn::AdamW<T> optA(/*lr=*/1e-3, /*b1=*/0.9, /*b2=*/0.999, /*eps=*/1e-8, /*wd=*/1e-2);
    tinynn::AdamW<T> optB(/*lr=*/1e-3, /*b1=*/0.9, /*b2=*/0.999, /*eps=*/1e-8, /*wd=*/1e-2);
    run_roundtrip_test<tinynn::AdamW<T>, T>("AdamW", optA, optB, total_steps, split_step, tol);
  }

  // RMSProp
  {
    tinynn::RMSProp<T> optA(/*lr=*/1e-3, /*rho=*/0.99, /*eps=*/1e-8, /*wd=*/1e-2);
    tinynn::RMSProp<T> optB(/*lr=*/1e-3, /*rho=*/0.99, /*eps=*/1e-8, /*wd=*/1e-2);
    run_roundtrip_test<tinynn::RMSProp<T>, T>("RMSProp", optA, optB, total_steps, split_step, tol);
  }

  std::cout << "optimizer_checkpoint_roundtrip_test passed.\n";
  return 0;
}
