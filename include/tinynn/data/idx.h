#pragma once

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinynn::idx {

inline uint32_t ReadU32BE(std::ifstream& in) {
  uint8_t b[4]{};
  in.read(reinterpret_cast<char*>(b), 4);
  if (!in) {
    throw std::runtime_error("idx: failed to read u32");
  }
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
         (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

inline int32_t ReadI32BE(std::ifstream& in) {
  const uint32_t u = ReadU32BE(in);
  return static_cast<int32_t>(u);
}

inline std::vector<int32_t> ReadLabelsInt32(const std::string& path,
                                            uint32_t* out_count = nullptr,
                                            uint32_t* out_dim1 = nullptr) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("idx: cannot open labels file: " + path);

  const uint32_t magic = ReadU32BE(in);

  if (magic == 2049u) {
    // ubyte, 1D: [N]
    const uint32_t count = ReadU32BE(in);
    std::vector<uint8_t> tmp(count);
    in.read(reinterpret_cast<char*>(tmp.data()),
            static_cast<std::streamsize>(count));
    if (!in)
      throw std::runtime_error("idx: failed to read ubyte labels payload");

    std::vector<int32_t> labels(count);
    for (uint32_t i = 0; i < count; ++i)
      labels[i] = static_cast<int32_t>(tmp[i]);
    if (out_count) *out_count = count;
    if (out_dim1) *out_dim1 = 1;
    return labels;
  }

  if (magic == 3073u) {
    // int32, 1D: [N]
    const uint32_t count = ReadU32BE(in);
    std::vector<int32_t> labels(count);
    for (uint32_t i = 0; i < count; ++i) labels[i] = ReadI32BE(in);
    if (out_count) *out_count = count;
    if (out_dim1) *out_dim1 = 1;
    return labels;
  }

  if (magic == 3074u) {
    // int32, 2D: [N, K] (QMNIST uses K = 8)
    const uint32_t n = ReadU32BE(in);
    const uint32_t k = ReadU32BE(in);
    if (k == 0) throw std::runtime_error("idx: labels dim1 is 0");

    std::vector<int32_t> labels(n);

    for (uint32_t i = 0; i < n; ++i) {
      // col0: class label (0..9)
      labels[i] = ReadI32BE(in);

      // skip remaining columns
      for (uint32_t j = 1; j < k; ++j) {
        (void)ReadI32BE(in);
      }
    }

    if (out_count) *out_count = n;
    if (out_dim1) *out_dim1 = k;
    return labels;
  }

  throw std::runtime_error("idx: labels magic mismatch (unsupported): " +
                           std::to_string(magic));
}

inline std::vector<uint8_t> ReadLabels(const std::string& path, uint32_t* out_count = nullptr) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("idx: cannot open labels file: " + path);

  const uint32_t magic = ReadU32BE(in);
  if (magic != 3074u) throw std::runtime_error("idx: labels magic mismatch");

  const uint32_t count = ReadU32BE(in);
  std::vector<uint8_t> labels(count);
  in.read(reinterpret_cast<char*>(labels.data()), static_cast<std::streamsize>(count));
  if (!in) throw std::runtime_error("idx: failed to read labels payload");

  if (out_count) *out_count = count;
  return labels;
}

inline std::vector<uint8_t> ReadImages(const std::string& path,
                                       uint32_t* out_count = nullptr,
                                       uint32_t* out_rows = nullptr,
                                       uint32_t* out_cols = nullptr) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("idx: cannot open images file: " + path);

  const uint32_t magic = ReadU32BE(in);
  if (magic != 2051u) throw std::runtime_error("idx: images magic mismatch");

  const uint32_t count = ReadU32BE(in);
  const uint32_t rows = ReadU32BE(in);
  const uint32_t cols = ReadU32BE(in);

  const uint64_t total = uint64_t(count) * uint64_t(rows) * uint64_t(cols);
  if (total > uint64_t(std::numeric_limits<size_t>::max()))
    throw std::runtime_error("idx: images too large");

  std::vector<uint8_t> images(static_cast<size_t>(total));
  in.read(reinterpret_cast<char*>(images.data()), static_cast<std::streamsize>(images.size()));
  if (!in) throw std::runtime_error("idx: failed to read images payload");

  if (out_count) *out_count = count;
  if (out_rows) *out_rows = rows;
  if (out_cols) *out_cols = cols;
  return images;
}

}  // namespace tinynn::idx
