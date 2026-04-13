#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tinynn/nn/init_policy.h>
#include <tinynn/nn/layer.h>
#include <tinynn/nn/layers/affine.h>
#include <tinynn/nn/layers/conv2d.h>
#include <tinynn/nn/parameter_view.h>

namespace {

template <class T>
std::vector<T> snapshot_params(tinynn::Layer<T>& layer) {
  std::vector<tinynn::ParameterView<T>> pvs;
  layer.collect_parameter_views(pvs);

  std::vector<T> flat;
  for (auto& pv : pvs) {
    auto p = pv.param;
    for (tinynn::SizeType i = 0; i < p.size(); ++i) {
      flat.push_back(p[i]);
    }
  }
  return flat;
}

template <class T>
bool exact_equal(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) {
      return false;
    }
  }
  return true;
}

template <class T>
bool all_zero(const std::vector<T>& v) {
  for (const auto& x : v) {
    if (x != static_cast<T>(0)) {
      return false;
    }
  }
  return true;
}

template <class T>
void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error("[NG] " + msg);
  }
}

}  // namespace

int main() {
  using T = double;

  try {
    // -------------------------
    // Affine
    // -------------------------
    tinynn::Layer<T>::reset_param_id_allocator_for_testing();

    tinynn::Affine<T> affine(8, 4, true);

    // same policy + same seed => exact same params
    affine.set_init_policy(tinynn::InitPolicy::kHeNormal);
    affine.set_init_seed(123);
    affine.reset_parameters();
    const auto affine_he_1 = snapshot_params(affine);

    affine.set_init_policy(tinynn::InitPolicy::kHeNormal);
    affine.set_init_seed(123);
    affine.reset_parameters();
    const auto affine_he_2 = snapshot_params(affine);

    require<T>(exact_equal(affine_he_1, affine_he_2),
               "Affine: same seed/policy should reproduce identical params");

    // different policy => should differ
    affine.set_init_policy(tinynn::InitPolicy::kXavierUniform);
    affine.set_init_seed(123);
    affine.reset_parameters();
    const auto affine_xavier = snapshot_params(affine);

    require<T>(!exact_equal(affine_he_1, affine_xavier),
               "Affine: HeNormal and XavierUniform should differ");

    // zeros policy => all params zero
    affine.set_init_policy(tinynn::InitPolicy::kZeros);
    affine.reset_parameters();
    const auto affine_zero = snapshot_params(affine);

    require<T>(all_zero(affine_zero),
               "Affine: Zeros policy should zero all params");

    // -------------------------
    // Conv2d
    // -------------------------
    tinynn::Layer<T>::reset_param_id_allocator_for_testing();

    tinynn::Conv2d<T> conv(tinynn::Conv2dOptions{
        .in_channels = 3,
        .out_channels = 4,
        .kernel_h = 3,
        .kernel_w = 3,
        .bias = true,
        .stride_h = 1,
        .stride_w = 1,
        .pad_h = 1,
        .pad_w = 1,
        .dilation_h = 1,
        .dilation_w = 1,
        .groups = 1,
    });

    // same policy + same seed => exact same params
    conv.set_init_policy(tinynn::InitPolicy::kHeNormal);
    conv.set_init_seed(777);
    conv.reset_parameters();
    const auto conv_he_1 = snapshot_params(conv);

    conv.set_init_policy(tinynn::InitPolicy::kHeNormal);
    conv.set_init_seed(777);
    conv.reset_parameters();
    const auto conv_he_2 = snapshot_params(conv);

    require<T>(exact_equal(conv_he_1, conv_he_2),
               "Conv2d: same seed/policy should reproduce identical params");

    // different policy => should differ
    conv.set_init_policy(tinynn::InitPolicy::kXavierUniform);
    conv.set_init_seed(777);
    conv.reset_parameters();
    const auto conv_xavier = snapshot_params(conv);

    require<T>(!exact_equal(conv_he_1, conv_xavier),
               "Conv2d: HeNormal and XavierUniform should differ");

    // zeros policy => all params zero
    conv.set_init_policy(tinynn::InitPolicy::kZeros);
    conv.reset_parameters();
    const auto conv_zero = snapshot_params(conv);

    require<T>(all_zero(conv_zero),
               "Conv2d: Zeros policy should zero all params");

    std::cout << "[OK] init policy smoke test passed (Affine / Conv2d)\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
