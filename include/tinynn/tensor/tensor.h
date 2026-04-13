#pragma once

#include <cassert>
#include <span>
#include <stdexcept>
#include <vector>

#include <tinynn/tensor/shape.h>
#include <tinynn/tensor/tensor_view.h>

namespace tinynn {

// Minimal owning tensor (contiguous storage).
template <class T>
class Tensor {
 public:
  using ValueType = T;

  Tensor() = default;

  explicit Tensor(const Shape& shape)
    : shape_(shape), data_(shape.volume()) {}

  Tensor(const Shape& shape, const T& fill_value)
    : shape_(shape), data_(shape.volume(), fill_value) {}

  explicit Tensor(std::initializer_list<SizeType> dims)
      : shape_(Shape{dims}), data_(shape_.volume()) {}

  Tensor(std::initializer_list<SizeType> dims, const T& fill_value)
      : shape_(Shape{dims}), data_(shape_.volume(), fill_value) {}

  const Shape& shape() const noexcept { return shape_; }

  SizeType size() const noexcept { return data_.size(); }

  bool empty() const noexcept { return data_.empty(); }

  T* data() noexcept { return data_.data(); }

  const T* data() const noexcept { return data_.data(); }

  // Non-owning views
  [[nodiscard]] TensorView<T> view() noexcept {
    return TensorView<T>(shape_, data_.empty() ? nullptr : data_.data());
  }

  [[nodiscard]] ConstTensorView<T> view() const noexcept {
    return ConstTensorView<T>(shape_, data_.empty() ? nullptr : data_.data());
  }

  // Returns a reshaped view (Tensor itself is not modified).
  [[nodiscard]] TensorView<T> reshaped_view(const Shape& new_shape) {
    return view().reshaped(new_shape);
  }

  [[nodiscard]] ConstTensorView<T> reshaped_view(const Shape& new_shape) const {
    return view().reshaped(new_shape);
  }

  [[nodiscard]] TensorView<T> reshaped_view(std::initializer_list<SizeType> dims) {
    return view().reshaped(dims);
  }

  [[nodiscard]] ConstTensorView<T> reshaped_view(
      std::initializer_list<SizeType> dims) const {
    return view().reshaped(dims);
  }

  [[nodiscard]] TensorView<T> reshaped_view(std::span<const SizeType> dims) {
    return view().reshaped(dims);
  }

  [[nodiscard]] ConstTensorView<T> reshaped_view(
    std::span<const SizeType> dims) const {
    return view().reshaped(dims);
  }

  T& operator[](SizeType i) noexcept { return data_[i]; }

  const T& operator[](SizeType i) const noexcept { return data_[i]; }

  T& at(SizeType i) {
    if (i >= data_.size()) {
      throw std::out_of_range("tinynn::Tensor::at: index out of range");
    }
    return data_[i];
  }

  const T& at(SizeType i) const {
    if (i >= data_.size()) {
      throw std::out_of_range("tinynn::Tensor::at: index out of range");
    }
    return data_[i];
  }

  T& at(SizeType r, SizeType c) {
    EnsureRank2();
    const SizeType cols = shape_.cols();
    if (r >= shape_.rows() || c >= cols) {
      throw std::out_of_range("tinynn::Tensor::at(r,c): index out of range");
    }
    return data_[r * cols + c];
  }

  const T& at(SizeType r, SizeType c) const {
    EnsureRank2();
    const SizeType cols = shape_.cols();
    if (r >= shape_.rows() || c >= cols) {
      throw std::out_of_range("tinynn::Tensor::at(r,c): index out of range");
    }
    return data_[r * cols + c];
  }

  // Fast 2D access (no bounds check). Row-major contiguous.
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

  // Returns a pointer to the first element of row r (rank must be 2).
  T* row_ptr(SizeType r) noexcept {
    assert(shape_.rank() == 2);
    assert(r < shape_.dim_unchecked(0));
    const SizeType cols = shape_.dim_unchecked(1);
    return data_.data() + (r * cols);
  }

  const T* row_ptr(SizeType r) const noexcept {
    assert(shape_.rank() == 2);
    assert(r < shape_.dim_unchecked(0));
    const SizeType cols = shape_.dim_unchecked(1);
    return data_.data() + (r * cols);
  }

  std::span<T> row_span(SizeType r) noexcept {
    assert(shape_.rank() == 2);
    assert(r < shape_.dim_unchecked(0));
    const SizeType cols = shape_.dim_unchecked(1);
    return std::span<T>(data_.data() + r * cols, cols);
  }

  std::span<const T> row_span(SizeType r) const noexcept {
    assert(shape_.rank() == 2);
    assert(r < shape_.dim_unchecked(0));
    const SizeType cols = shape_.dim_unchecked(1);
    return std::span<const T>(data_.data() + r * cols, cols);
  }

  SizeType row_count() const noexcept {
    assert(shape_.rank() == 2);
    return shape_.dim_unchecked(0);
  }

  SizeType col_count() const noexcept {
    assert(shape_.rank() == 2);
    return shape_.dim_unchecked(1);
  }

  // Fast 3D access (no bounds check). Row-major contiguous.
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

  T& at(SizeType a, SizeType b, SizeType c) {
    EnsureRank3();
    if (a >= shape_.dim(0) || b >= shape_.dim(1) || c >= shape_.dim(2)) {
      throw std::out_of_range("tinynn::Tensor::at(a,b,c): index out of range");
    }
    return data_[idx3_(a, b, c, shape_.dim_unchecked(1),
                       shape_.dim_unchecked(2))];
  }

  const T& at(SizeType a, SizeType b, SizeType c) const {
    EnsureRank3();
    if (a >= shape_.dim(0) || b >= shape_.dim(1) || c >= shape_.dim(2)) {
      throw std::out_of_range("tinynn::Tensor::at(a,b,c): index out of range");
    }
    return data_[idx3_(a, b, c, shape_.dim_unchecked(1),
                       shape_.dim_unchecked(2))];
  }

  // Fast 4D access (no bounds check). Row-major contiguous.
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

  T& at(SizeType a, SizeType b, SizeType c, SizeType d) {
    EnsureRank4();
    if (a >= shape_.dim(0) || b >= shape_.dim(1) || c >= shape_.dim(2) ||
        d >= shape_.dim(3)) {
      throw std::out_of_range(
          "tinynn::Tensor::at(a,b,c,d): index out of range");
    }
    return data_[idx4_(a, b, c, d, shape_.dim_unchecked(1),
                       shape_.dim_unchecked(2), shape_.dim_unchecked(3))];
  }

  const T& at(SizeType a, SizeType b, SizeType c, SizeType d) const {
    EnsureRank4();
    if (a >= shape_.dim(0) || b >= shape_.dim(1) || c >= shape_.dim(2) ||
        d >= shape_.dim(3)) {
      throw std::out_of_range(
          "tinynn::Tensor::at(a,b,c,d): index out of range");
    }
    return data_[idx4_(a, b, c, d, shape_.dim_unchecked(1),
                       shape_.dim_unchecked(2), shape_.dim_unchecked(3))];
  }

  // Reshape without reallocating storage. Total element count must match.
  void reshape(const Shape& new_shape) {
    if (new_shape.volume() != shape_.volume()) {
      throw std::invalid_argument("tinynn::Tensor::reshape: volume mismatch");
    }
    shape_ = new_shape;
  }

 private:

  void EnsureRank2() const {
    if (shape_.rank() != 2) {
      throw std::invalid_argument("tinynn::Tensor: rank must be 2 for at(r,c)");
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
      throw std::invalid_argument("tinynn::Tensor: rank must be 3");
    }
  }

  void EnsureRank4() const {
    if (shape_.rank() != 4) {
      throw std::invalid_argument("tinynn::Tensor: rank must be 4");
    }
  }

  Shape shape_{};
  std::vector<T> data_{};
};

}  // namespace tinynn
