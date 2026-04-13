#pragma once

#include <vector>

#include <tinynn/tensor/tensor.h>

namespace tinynn {

// Mini-batch container: x = inputs, y = class indices.
template <class T>
struct Batch {
  Tensor<T> x;
  std::vector<SizeType> y;

  SizeType batch_size() const noexcept { return static_cast<SizeType>(y.size()); }
};

}  // namespace tinynn
