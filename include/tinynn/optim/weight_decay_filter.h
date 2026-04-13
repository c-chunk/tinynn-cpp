#pragma once

#include <tinynn/nn/parameter_view.h>

namespace tinynn {

// Apply weight decay only to weight-like tensors (not bias / rank-1 params).
template <class T>
[[nodiscard]] inline bool should_apply_weight_decay(const ParameterView<T>& pv) {
  if (pv.kind == ParamKind::kBias) return false;
  if (pv.param.shape().rank() == 1) return false;

  return true;
}

}  // namespace tinynn
