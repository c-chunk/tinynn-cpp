#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tinynn/nn/parameter_view.h>
#include <tinynn/nn/sequential.h>

namespace tinynn {

// Load parameters from "TNYNCKPT" binary into a model.
// Preconditions:
// - model.collect_parameter_views(out) returns the same params in the same order
//   as the checkpoint was saved with.
// - T matches the checkpoint element type (e.g., float vs double).
template <class T>
inline void load_checkpoint_params(const std::string& path,
                                   Sequential<T>& model) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("load_checkpoint_params: cannot open file: " +
                             path);

  auto read_u8 = [&]() -> uint8_t {
    uint8_t v{};
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("load_checkpoint_params: read_u8 failed");
    return v;
  };
  auto read_u32 = [&]() -> uint32_t {
    uint32_t v{};
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in)
      throw std::runtime_error("load_checkpoint_params: read_u32 failed");
    return v;
  };
  auto read_u64 = [&]() -> uint64_t {
    uint64_t v{};
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in)
      throw std::runtime_error("load_checkpoint_params: read_u64 failed");
    return v;
  };

  // magic
  char magic[8]{};
  in.read(magic, sizeof(magic));
  if (!in)
    throw std::runtime_error("load_checkpoint_params: read magic failed");
  const char expected[8] = {'T', 'N', 'Y', 'N', 'C', 'K', 'P', 'T'};
  for (int i = 0; i < 8; ++i) {
    if (magic[i] != expected[i]) {
      throw std::runtime_error("load_checkpoint_params: magic mismatch");
    }
  }

  const uint32_t version = read_u32();
  if (version != 1u && version != 2u) {
    throw std::runtime_error("load_checkpoint_params: unsupported version");
  }

  if (version == 2u) {
    const uint32_t type_size = read_u32();
    if (type_size != static_cast<uint32_t>(sizeof(T))) {
      throw std::runtime_error("load_checkpoint_params: type_size mismatch");
    }
  }

  const uint64_t file_num_params = read_u64();

  std::vector<ParameterView<T>> params;
  model.collect_parameter_views(params);
  if (file_num_params != static_cast<uint64_t>(params.size())) {
    throw std::runtime_error("load_checkpoint_params: num_params mismatch");
  }

  for (size_t i = 0; i < params.size(); ++i) {
    auto& pv = params[i];

    // ParamKind strict
    const uint8_t file_kind_u8 = read_u8();
    const uint8_t expected_kind_u8 = static_cast<uint8_t>(pv.kind);
    if (file_kind_u8 != expected_kind_u8) {
      throw std::runtime_error(
          "load_checkpoint_params: ParamKind mismatch at param " +
          std::to_string(i));
    }

    const uint32_t rank = read_u32();
    if (rank != static_cast<uint32_t>(pv.param.shape().rank())) {
      throw std::runtime_error(
          "load_checkpoint_params: rank mismatch at param " +
          std::to_string(i));
    }
    for (uint32_t d = 0; d < rank; ++d) {
      const uint64_t dim = read_u64();
      const uint64_t expected_dim = static_cast<uint64_t>(
          pv.param.shape().dim_unchecked(static_cast<SizeType>(d)));
      if (dim != expected_dim) {
        throw std::runtime_error(
            "load_checkpoint_params: dim mismatch at param " +
            std::to_string(i));
      }
    }

    const uint64_t count = read_u64();
    const uint64_t expected_count = static_cast<uint64_t>(pv.param.size());
    if (count != expected_count) {
      throw std::runtime_error(
          "load_checkpoint_params: count mismatch at param " +
          std::to_string(i));
    }

    T* dst = pv.param.data();
    if (count > 0 && dst == nullptr) {
      throw std::runtime_error(
          "load_checkpoint_params: param data is null at param " +
          std::to_string(i));
    }
    in.read(reinterpret_cast<char*>(dst),
            static_cast<std::streamsize>(sizeof(T) * count));
    if (!in) {
      throw std::runtime_error(
          "load_checkpoint_params: read param bytes failed at param " +
          std::to_string(i));
    }
  }

  // ---- trailing bytes check ----
  {
    char extra = 0;
    in.read(&extra, 1);

    if (in.gcount() == 1) {
      throw std::runtime_error(
          "load_checkpoint_params: trailing bytes detected");
    }
    if (!in.eof()) {
      throw std::runtime_error(
          "load_checkpoint_params: failed to reach EOF cleanly");
    }
    in.clear();  // clear eof/fail flags
  }
}

template <class T>
inline void save_checkpoint_params(const std::string& path,
                                   Sequential<T>& model) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("save_checkpoint_params: cannot open file: " +
                             path);
  }

  auto write_u8 = [&](uint8_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    if (!out)
      throw std::runtime_error("save_checkpoint_params: write_u8 failed");
  };
  auto write_u32 = [&](uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    if (!out)
      throw std::runtime_error("save_checkpoint_params: write_u32 failed");
  };
  auto write_u64 = [&](uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    if (!out)
      throw std::runtime_error("save_checkpoint_params: write_u64 failed");
  };

  std::vector<ParameterView<T>> params;
  model.collect_parameter_views(params);

  // magic(8)
  const char magic[8] = {'T', 'N', 'Y', 'N', 'C', 'K', 'P', 'T'};
  out.write(magic, sizeof(magic));
  if (!out)
    throw std::runtime_error("save_checkpoint_params: write magic failed");

  // version=2 + type_size
  write_u32(2u);
  write_u32(static_cast<uint32_t>(sizeof(T)));

  // num_params
  write_u64(static_cast<uint64_t>(params.size()));

  // params
  for (size_t i = 0; i < params.size(); ++i) {
    const auto& pv = params[i];

    // kind (strictly loaded later)
    write_u8(static_cast<uint8_t>(pv.kind));

    // shape rank/dims
    const uint32_t rank = static_cast<uint32_t>(pv.param.shape().rank());
    write_u32(rank);
    for (uint32_t d = 0; d < rank; ++d) {
      const uint64_t dim = static_cast<uint64_t>(
          pv.param.shape().dim_unchecked(static_cast<SizeType>(d)));
      write_u64(dim);
    }

    // count + bytes
    const uint64_t count = static_cast<uint64_t>(pv.param.size());
    write_u64(count);

    const T* src = pv.param.data();
    if (count > 0 && src == nullptr) {
      throw std::runtime_error(
          "save_checkpoint_params: param data is null at param " +
          std::to_string(i));
    }
    out.write(reinterpret_cast<const char*>(src),
              static_cast<std::streamsize>(sizeof(T) * count));
    if (!out) {
      throw std::runtime_error(
          "save_checkpoint_params: write param bytes failed at param " +
          std::to_string(i));
    }
  }
}

}  // namespace tinynn
