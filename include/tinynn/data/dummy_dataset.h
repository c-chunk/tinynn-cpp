#pragma once

#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

#include <tinynn/data/batch.h>
#include <tinynn/tensor/tensor.h>

namespace tinynn {

// In-memory tiny dataset for smoke tests.
// Stores X as [N, D] and y as [N].
template <class T>
class DummyDataset {
 public:
  DummyDataset(Tensor<T> x_all, std::vector<SizeType> y_all)
      : x_all_(std::move(x_all)), y_all_(std::move(y_all)) {
    if (x_all_.shape().rank() != 2) {
      throw std::invalid_argument("DummyDataset: x_all must be rank-2 [N, D]");
    }
    const SizeType n = x_all_.row_count();
    if (static_cast<SizeType>(y_all_.size()) != n) {
      throw std::invalid_argument("DummyDataset: y_all size must match N");
    }
  }

  SizeType size() const noexcept { return x_all_.row_count(); }
  SizeType dim() const noexcept { return x_all_.col_count(); }

  // Fill out batch from given indices.
  void get_batch(std::span<const SizeType> indices, Batch<T>* out) const {
    if (out == nullptr) {
      throw std::invalid_argument("DummyDataset::get_batch: out is null");
    }
    const SizeType b = static_cast<SizeType>(indices.size());
    const SizeType d = dim();

    out->x = Tensor<T>(Shape{b, d});
    out->y.resize(b);

    auto xdst = out->x.view();
    auto xsrc = x_all_.view();

    for (SizeType bi = 0; bi < b; ++bi) {
      const SizeType idx = indices[bi];
      if (idx >= size()) {
        throw std::invalid_argument("DummyDataset::get_batch: index out of range");
      }
      out->y[bi] = y_all_[idx];

      // Copy one row.
      auto src_row = xsrc.row_span(idx);
      auto dst_row = xdst.row_span(bi);
      std::copy(src_row.begin(), src_row.end(), dst_row.begin());
    }
  }

 private:
  Tensor<T> x_all_;
  std::vector<SizeType> y_all_;
};

}  // namespace tinynn
