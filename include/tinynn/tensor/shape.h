#pragma once

#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tinynn/tinynn.h>

namespace tinynn {

// Minimal shape class.
class Shape {
 public:
  Shape() = default;

  explicit Shape(std::initializer_list<SizeType> dims) : dims_(dims) {
    Validate();
  }

  explicit Shape(std::vector<SizeType> dims) : dims_(std::move(dims)) {
    Validate();
  }

  SizeType rank() const noexcept { return dims_.size(); }

  const std::vector<SizeType>& dims() const noexcept { return dims_; }

  SizeType dim(SizeType i) const {
    if (i >= dims_.size()) {
      throw std::out_of_range("tinynn::Shape::dim: index out of range");
    }
    return dims_[i];
  }

  // Unchecked accessors (no exceptions). Preconditions must hold.
  SizeType dim_unchecked(SizeType i) const noexcept { return dims_[i]; }

  const SizeType* data() const noexcept { return dims_.data(); }

  // Total number of elements (0-rank shape has volume 1).
  SizeType volume() const noexcept {
    SizeType v = 1;
    for (SizeType d : dims_) {
      v *= d;
    }
    return v;
  }

  bool operator==(const Shape& other) const noexcept {
    return dims_ == other.dims_;
  }

  bool operator!=(const Shape& other) const noexcept {
    return !(*this == other);
  }

  SizeType rows() const {
    if (rank() != 2) {
      throw std::invalid_argument("tinynn::Shape::rows: rank must be 2");
    }
    return dims_[0];
  }

  SizeType cols() const {
    if (rank() != 2) {
      throw std::invalid_argument("tinynn::Shape::cols: rank must be 2");
    }
    return dims_[1];
  }

 private:
  void Validate() const {
    // Minimal validation: this shape class currently allows empty shapes
    // and zero-length dimensions.
  }

  std::vector<SizeType> dims_;
};

}  // namespace tinynn
