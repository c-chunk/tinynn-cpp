#pragma once

#include <tinynn/tensor/tensor_view.h>

#include <cstdint>

namespace tinynn {

template <class T>
struct BufferView {
  TensorView<T> buf;
  std::uintptr_t id = 0;
};

}  // namespace tinynn
