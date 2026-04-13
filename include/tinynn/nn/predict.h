#pragma once

#include <stdexcept>
#include <vector>

#include <tinynn/nn/sequential.h>

namespace tinynn {

// logits: shape [B, C]
template <class T>
std::vector<SizeType> argmax_by_row(ConstTensorView<T> logits) {
  const auto& s = logits.shape();
  if (s.rank() != 2) {
    throw std::invalid_argument(
        "tinynn::argmax_by_row: logits must be rank-2 [B, C]");
  }

  const SizeType batch = logits.row_count();
  const SizeType classes = logits.col_count();
  if (classes == 0) {
    throw std::invalid_argument(
        "tinynn::argmax_by_row: class dimension is zero");
  }

  std::vector<SizeType> pred(static_cast<std::size_t>(batch));

  for (SizeType r = 0; r < batch; ++r) {
    auto row = logits.row_span(r);

    SizeType best = 0;
    T best_val = row[0];
    for (SizeType c = 1; c < classes; ++c) {
      if (row[c] > best_val) {
        best_val = row[c];
        best = c;
      }
    }
    pred[static_cast<std::size_t>(r)] = best;
  }
  return pred;
}

template <class T>
std::vector<SizeType> argmax_by_row(TensorView<T> logits) {
  return argmax_by_row<T>(ConstTensorView<T>(logits));
}

template <class T>
std::vector<SizeType> predict_label(Sequential<T>& model,
                                    ConstTensorView<T> x) {
  auto logits = model.forward(x);   // TensorView<T>
  return argmax_by_row<T>(logits);  // TensorView overload
}

}  // namespace tinynn
