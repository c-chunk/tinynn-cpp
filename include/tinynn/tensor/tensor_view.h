#pragma once

#include <cassert>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <tinynn/tensor/shape.h>

namespace tinynn {

// Non-owning view to contiguous row-major tensor storage.
template <class T>
class TensorView {
 public:
  using ValueType = T;
  using NonConstT = std::remove_const_t<T>;
  using ConstView = TensorView<const NonConstT>;

  // Implicit conversion: TensorView<T> -> TensorView<const T>
  operator ConstView() const noexcept { return ConstView(shape_, data_); }

  // (Optional) explicit helper
  [[nodiscard]] ConstView as_const() const noexcept {
    return ConstView(shape_, data_);
  }

  TensorView() = default;

  TensorView(const Shape& shape, T* data) : shape_(shape), data_(data) {
    // nullptr is allowed only for empty views.
  }

  const Shape& shape() const noexcept { return shape_; }

  SizeType size() const noexcept { return shape_.volume(); }

  bool empty() const noexcept { return size() == 0; }

  T* data() noexcept { return data_; }

  const T* data() const noexcept { return data_; }

  // 1D access (no bounds check).
  T& operator[](SizeType i) noexcept { return data_[i]; }

  const T& operator[](SizeType i) const noexcept { return data_[i]; }

  // 1D access (bounds check).
  T& at(SizeType i) {
    if (i >= size()) {
      throw std::out_of_range("tinynn::TensorView::at: index out of range");
    }
    return data_[i];
  }

  const T& at(SizeType i) const {
    if (i >= size()) {
      throw std::out_of_range("tinynn::TensorView::at: index out of range");
    }
    return data_[i];
  }

  // ---- reshape (view) ----
  // In-place reshape: changes only the view's shape
  // (data pointer is unchanged).
  void reshape_inplace(const Shape& new_shape) {
    if (new_shape.volume() != shape_.volume()) {
      throw std::invalid_argument(
          "tinynn::TensorView::reshape_inplace: volume mismatch");
    }
    shape_ = new_shape;
  }

  // Returns a new view with a different shape (volume must match).
  [[nodiscard]] TensorView reshaped(const Shape& new_shape) const {
    TensorView v = *this;
    v.reshape_inplace(new_shape);
    return v;
  }

  // Convenience overload: initializer_list
  [[nodiscard]] TensorView reshaped(std::initializer_list<SizeType> dims) const {
    return reshaped(Shape{dims});
  }

  // Convenience overload: span
  [[nodiscard]] TensorView reshaped(std::span<const SizeType> dims) const {
    // Shape takes vector, so copy dims into vector.
    return reshaped(Shape{std::vector<SizeType>(dims.begin(), dims.end())});
  }

  // ---- Rank-2 convenience API (row-major) ----
  SizeType row_count() const noexcept {
    assert(shape_.rank() == 2);
    return shape_.dim_unchecked(0);
  }

  SizeType col_count() const noexcept {
    assert(shape_.rank() == 2);
    return shape_.dim_unchecked(1);
  }

  T* row_ptr(SizeType r) noexcept {
    assert(shape_.rank() == 2);
    assert(r < shape_.dim_unchecked(0));
    const SizeType cols = shape_.dim_unchecked(1);
    return data_ + (r * cols);
  }

  const T* row_ptr(SizeType r) const noexcept {
    assert(shape_.rank() == 2);
    assert(r < shape_.dim_unchecked(0));
    const SizeType cols = shape_.dim_unchecked(1);
    return data_ + (r * cols);
  }

  [[nodiscard]] std::span<T> row_span(SizeType r) noexcept {
    assert(shape_.rank() == 2);
    assert(r < shape_.dim_unchecked(0));
    const SizeType cols = shape_.dim_unchecked(1);
    return std::span<T>(data_ + r * cols, cols);
  }

  [[nodiscard]] std::span<const T> row_span(SizeType r) const noexcept {
    assert(shape_.rank() == 2);
    assert(r < shape_.dim_unchecked(0));
    const SizeType cols = shape_.dim_unchecked(1);
    return std::span<const T>(data_ + r * cols, cols);
  }

  // Fast 2D access (no bounds check).
  T& operator()(SizeType r, SizeType c) noexcept {
    assert(shape_.rank() == 2);
    assert(r < shape_.dim_unchecked(0));
    assert(c < shape_.dim_unchecked(1));
    const SizeType cols = shape_.dim_unchecked(1);
    return data_[r * cols + c];
  }

  const T& operator()(SizeType r, SizeType c) const noexcept {
    assert(shape_.rank() == 2);
    assert(r < shape_.dim_unchecked(0));
    assert(c < shape_.dim_unchecked(1));
    const SizeType cols = shape_.dim_unchecked(1);
    return data_[r * cols + c];
  }

  // Safe 2D access (bounds check).
  T& at(SizeType r, SizeType c) {
    EnsureRank2();
    if (r >= shape_.rows() || c >= shape_.cols()) {
      throw std::out_of_range("tinynn::TensorView::at(r,c): index out of range");
    }
    const SizeType cols = shape_.cols();
    return data_[r * cols + c];
  }

  const T& at(SizeType r, SizeType c) const {
    EnsureRank2();
    if (r >= shape_.rows() || c >= shape_.cols()) {
      throw std::out_of_range("tinynn::TensorView::at(r,c): index out of range");
    }
    const SizeType cols = shape_.cols();
    return data_[r * cols + c];
  }

  // ---- Rank-3 convenience API (row-major) ----
  // Fast 3D access (no bounds check).
  T& operator()(SizeType a, SizeType b, SizeType c) noexcept {
    assert(shape_.rank() == 3);
    assert(a < shape_.dim_unchecked(0));
    assert(b < shape_.dim_unchecked(1));
    assert(c < shape_.dim_unchecked(2));
    return data_[idx3_(a, b, c, shape_.dim_unchecked(1),
                       shape_.dim_unchecked(2))];
  }

  const T& operator()(SizeType a, SizeType b, SizeType c) const noexcept {
    assert(shape_.rank() == 3);
    assert(a < shape_.dim_unchecked(0));
    assert(b < shape_.dim_unchecked(1));
    assert(c < shape_.dim_unchecked(2));
    return data_[idx3_(a, b, c, shape_.dim_unchecked(1),
                       shape_.dim_unchecked(2))];
  }

  // Safe 3D access (bounds check).
  T& at(SizeType a, SizeType b, SizeType c) {
    EnsureRank3();
    if (a >= shape_.dim(0) || b >= shape_.dim(1) || c >= shape_.dim(2)) {
      throw std::out_of_range(
          "tinynn::TensorView::at(a,b,c): index out of range");
    }
    return data_[idx3_(a, b, c, shape_.dim_unchecked(1),
                       shape_.dim_unchecked(2))];
  }

  const T& at(SizeType a, SizeType b, SizeType c) const {
    EnsureRank3();
    if (a >= shape_.dim(0) || b >= shape_.dim(1) || c >= shape_.dim(2)) {
      throw std::out_of_range(
          "tinynn::TensorView::at(a,b,c): index out of range");
    }
    return data_[idx3_(a, b, c, shape_.dim_unchecked(1),
                       shape_.dim_unchecked(2))];
  }

  // ---- Rank-4 convenience API (row-major) ----
  // Fast 4D access (no bounds check).
  T& operator()(SizeType a, SizeType b, SizeType c, SizeType d) noexcept {
    assert(shape_.rank() == 4);
    assert(a < shape_.dim_unchecked(0));
    assert(b < shape_.dim_unchecked(1));
    assert(c < shape_.dim_unchecked(2));
    assert(d < shape_.dim_unchecked(3));
    return data_[idx4_(a, b, c, d, shape_.dim_unchecked(1),
                       shape_.dim_unchecked(2), shape_.dim_unchecked(3))];
  }

  const T& operator()(SizeType a, SizeType b, SizeType c,
                      SizeType d) const noexcept {
    assert(shape_.rank() == 4);
    assert(a < shape_.dim_unchecked(0));
    assert(b < shape_.dim_unchecked(1));
    assert(c < shape_.dim_unchecked(2));
    assert(d < shape_.dim_unchecked(3));
    return data_[idx4_(a, b, c, d, shape_.dim_unchecked(1),
                       shape_.dim_unchecked(2), shape_.dim_unchecked(3))];
  }

  // Safe 4D access (bounds check).
  T& at(SizeType a, SizeType b, SizeType c, SizeType d) {
    EnsureRank4();
    if (a >= shape_.dim(0) || b >= shape_.dim(1) || c >= shape_.dim(2) ||
        d >= shape_.dim(3)) {
      throw std::out_of_range(
          "tinynn::TensorView::at(a,b,c,d): index out of range");
    }
    return data_[idx4_(a, b, c, d, shape_.dim_unchecked(1),
                       shape_.dim_unchecked(2), shape_.dim_unchecked(3))];
  }

  const T& at(SizeType a, SizeType b, SizeType c, SizeType d) const {
    EnsureRank4();
    if (a >= shape_.dim(0) || b >= shape_.dim(1) || c >= shape_.dim(2) ||
        d >= shape_.dim(3)) {
      throw std::out_of_range(
          "tinynn::TensorView::at(a,b,c,d): index out of range");
    }
    return data_[idx4_(a, b, c, d, shape_.dim_unchecked(1),
                       shape_.dim_unchecked(2), shape_.dim_unchecked(3))];
  }

 private:
  void EnsureRank2() const {
    if (shape_.rank() != 2) {
      throw std::invalid_argument("tinynn::TensorView: rank must be 2");
    }
  }

  static SizeType idx3_(SizeType a, SizeType b, SizeType c, SizeType B,
                        SizeType C) noexcept {
    return (a * B + b) * C + c;
  }

  static SizeType idx4_(SizeType a, SizeType b, SizeType c, SizeType d,
                        SizeType B, SizeType C, SizeType D) noexcept {
    return ((a * B + b) * C + c) * D + d;
  }

  void EnsureRank3() const {
    if (shape_.rank() != 3) {
      throw std::invalid_argument("tinynn::TensorView: rank must be 3");
    }
  }

  void EnsureRank4() const {
    if (shape_.rank() != 4) {
      throw std::invalid_argument("tinynn::TensorView: rank must be 4");
    }
  }

  Shape shape_{};
  T* data_ = nullptr;
};

// Convenience aliases
template <class T>
using ConstTensorView = TensorView<const T>;

}  // namespace tinynn
