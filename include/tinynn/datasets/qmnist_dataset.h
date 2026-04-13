#pragma once

#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tinynn/data/batch.h>
#include <tinynn/data/idx.h>
#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor.h>

namespace tinynn {

// QMNIST dataset for MLP-style input.
// Produces x with shape [B, rows * cols], normalized to [0, 1],
// and y with shape [B].
template <class T>
class QmnistDataset {
 public:
  struct Files {
    std::string images_path;
    std::string labels_path;
  };

  explicit QmnistDataset(const Files& files) {
    uint32_t n_img = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t n_lbl = 0;
    images_u8_ = idx::ReadImages(files.images_path, &n_img, &rows, &cols);
    labels_i32_ = idx::ReadLabelsInt32(files.labels_path, &n_lbl);

    if (n_img != n_lbl) {
      throw std::invalid_argument(
        "QmnistDataset: image/label count mismatch");
    }
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument("QmnistDataset: invalid image shape");
    }

    n_ = static_cast<SizeType>(n_img);
    rows_ = static_cast<SizeType>(rows);
    cols_ = static_cast<SizeType>(cols);
    image_size_ = rows_ * cols_;
  }

  [[nodiscard]] SizeType size() const noexcept { return n_; }
  [[nodiscard]] SizeType rows() const noexcept { return rows_; }
  [[nodiscard]] SizeType cols() const noexcept { return cols_; }
  [[nodiscard]] SizeType flat_dim() const noexcept { return image_size_; }

  // Fill Batch<T>:
  //   x = [B, rows*cols] float normalized to [0,1]
  //   y = [B]
  void get_batch(std::span<const SizeType> indices, Batch<T>* out) const {
    if (!out) {
      throw std::invalid_argument("QmnistDataset::get_batch: out is null");
    }

    const SizeType b = static_cast<SizeType>(indices.size());
    out->x = Tensor<T>(Shape{b, image_size_});
    out->y.resize(b);

    auto xdst = out->x.view();

    for (SizeType bi = 0; bi < b; ++bi) {
      const SizeType idx = indices[bi];
      if (idx >= n_) {
        throw std::invalid_argument(
            "QmnistDataset::get_batch: index out of range");
      }

      // labels: int32 (from idx2-int)
      const int32_t lab = labels_i32_[static_cast<size_t>(idx)];
      if (lab < 0 || lab > 9) {
        throw std::runtime_error(
            "QmnistDataset::get_batch: label out of range: " +
            std::to_string(lab));
      }
      out->y[bi] = static_cast<SizeType>(lab);

      const size_t base =
          static_cast<size_t>(idx) * static_cast<size_t>(image_size_);
      auto row = xdst.row_span(bi);

      for (SizeType j = 0; j < image_size_; ++j) {
        const uint8_t px = images_u8_[base + static_cast<size_t>(j)];
        row[j] = static_cast<T>(px) / static_cast<T>(255);
      }
    }
  }

 private:
  std::vector<uint8_t> images_u8_;  // [N * rows * cols]
  std::vector<int32_t> labels_i32_; // [N]
  SizeType n_ = 0;
  SizeType rows_ = 0;
  SizeType cols_ = 0;
  SizeType image_size_ = 0;
};

}  // namespace tinynn
