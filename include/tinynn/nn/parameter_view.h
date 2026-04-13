#pragma once

#include <cstdint>

#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

enum class ParamKind { kWeight, kBias };

template <class T>
struct ParameterView {
  TensorView<T> param;
  TensorView<const T> grad;
  ParamKind kind = ParamKind::kWeight;
  std::uintptr_t id = 0;
};

}  // namespace tinynn
