#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <tinynn/data/batch.h>

namespace tinynn {

// Dataset concept:
//  - size() -> SizeType
//  - get_batch(indices, Batch<T>* out)
template <class T, class Dataset>
class DataLoader {
 public:
  using SizeType = std::size_t;

  DataLoader(Dataset& ds, SizeType batch_size, bool shuffle = true,
             std::uint64_t seed = 0)
      : ds_(ds),
        batch_size_(batch_size),
        shuffle_(shuffle),
        base_seed_(seed),
        rng_(seed) {
    if (batch_size_ == 0) {
      throw std::invalid_argument("tinynn::DataLoader: batch_size must be > 0");
    }
    reset_indices();
  }

  SizeType batch_size() const noexcept { return batch_size_; }

  void set_epoch(SizeType epoch) { epoch_ = epoch; }

  // Prepare indices/order for the current epoch.
  // Call this once at the beginning of each epoch (typically from Trainer).
  void prepare_epoch() {
    // If dataset size changed, rebuild indices.
    const SizeType n = static_cast<SizeType>(ds_.size());
    if (indices_.size() != n) {
      reset_indices();  // rebuild [0..n)
    }

    if (shuffle_) {
      const std::uint64_t epoch_seed =
          mix_seed(base_seed_, static_cast<std::uint64_t>(epoch_));
      rng_.seed(epoch_seed);
      std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
  }

  // ===== iterator =====
  struct Iterator {
    DataLoader* loader{};
    SizeType pos{}; // start index

    Batch<T> operator*() const { return loader->get_batch(pos); }

    Iterator& operator++() {
      pos += loader->batch_size_;
      return *this;
    }

    bool operator!=(const Iterator& other) const {
      return pos < other.pos;
    }
  };

  Iterator begin() {
    return Iterator{this, 0};
  }
  Iterator end() {
    auto n = static_cast<SizeType>(indices_.size());
    return Iterator{this, n};
  }

 private:
  void reset_indices() {
    indices_.resize(ds_.size());
    for (SizeType i = 0; i < indices_.size(); ++i) {
      indices_[i] = i;
    }
  }

  static std::uint64_t mix_seed(std::uint64_t a, std::uint64_t b) {
    // simple 64bit hash mix (splitmix64 style)
    std::uint64_t x = a ^ (b + 0x9e3779b97f4a7c15ULL);
    x ^= (x >> 30);
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= (x >> 27);
    x *= 0x94d049bb133111ebULL;
    x ^= (x >> 31);
    return x;
  }

  auto get_batch(SizeType start) {
    const SizeType n = static_cast<SizeType>(indices_.size());
    if (start >= n) {
      return Batch<T>{};
    }

    const SizeType bs = std::min(batch_size_, n - start);
    const std::span<const SizeType> batch_indices(indices_.data() + start, bs);

    Batch<T> out;
    ds_.get_batch(batch_indices, &out);
    return out;
  }

 private:
  Dataset& ds_;
  SizeType batch_size_;
  bool shuffle_;

  std::uint64_t base_seed_;
  SizeType epoch_ = 0;

  std::mt19937_64 rng_;
  std::vector<SizeType> indices_;
};

}  // namespace tinynn
