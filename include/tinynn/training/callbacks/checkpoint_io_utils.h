#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tinynn/training/callbacks/checkpoint_stream.h>

namespace tinynn {

// ---- scalar ----
template <class T>
inline void ckpt_write_scalar(CheckpointWriter& w, const T& v, const char* what) {
  static_assert(std::is_trivially_copyable_v<T>);
  w.write_pod(v, what);
}

template <class T>
inline T ckpt_read_scalar(CheckpointReader& r, const char* what) {
  static_assert(std::is_trivially_copyable_v<T>);
  return r.read_pod<T>(what);
}

// ---- vector<T> ----
template <class T>
inline void ckpt_write_vector(CheckpointWriter& w,
                              const std::vector<T>& v,
                              const char* what = "vector") {
  static_assert(std::is_trivially_copyable_v<T>);
  w.write_u64(static_cast<std::uint64_t>(v.size()));
  if (!v.empty()) {
    w.write_bytes(v.data(), v.size() * sizeof(T), what);
  }
}

template <class T>
inline void ckpt_read_vector(CheckpointReader& r,
                             std::vector<T>& v,
                             const char* what = "vector") {
  static_assert(std::is_trivially_copyable_v<T>);
  const std::uint64_t n64 = r.read_u64();
  if (n64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(std::string("ckpt_read_vector: too large: ") + what);
  }
  const std::size_t n = static_cast<std::size_t>(n64);
  v.assign(n, T{});
  if (n) {
    r.read_bytes(v.data(), n * sizeof(T), what);
  }
}

// ---- unordered_map<id, vector<T>> ----
template <class T>
inline void ckpt_write_state_map(CheckpointWriter& w,
                                 const std::unordered_map<std::uintptr_t,
                                                          std::vector<T>>& mp,
                                 const char* what = "state_map") {
  (void)what;
  std::vector<std::uintptr_t> keys;
  keys.reserve(mp.size());
  for (const auto& kv : mp) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());

  w.write_u64(static_cast<std::uint64_t>(keys.size()));
  for (std::uintptr_t id : keys) {
    w.write_u64(static_cast<std::uint64_t>(id));
    const auto it = mp.find(id);
    if (it == mp.end()) {
      throw std::runtime_error("ckpt_write_state_map: missing key after sort");
    }
    ckpt_write_vector(w, it->second, "state_vec");
  }
}

template <class T>
inline void ckpt_read_state_map(CheckpointReader& r,
                                std::unordered_map<std::uintptr_t,
                                                   std::vector<T>>& mp,
                                const char* what = "state_map") {
  (void)what;
  mp.clear();
  const std::uint64_t n64 = r.read_u64();
  if (n64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("ckpt_read_state_map: too large");
  }
  const std::size_t n = static_cast<std::size_t>(n64);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uintptr_t id = static_cast<std::uintptr_t>(r.read_u64());
    std::vector<T> vec;
    ckpt_read_vector(r, vec, "state_vec");
    mp.emplace(id, std::move(vec));
  }
}

}  // namespace tinynn
