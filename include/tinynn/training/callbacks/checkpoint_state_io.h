#pragma once

#include <tinynn/nn/buffer_view.h>
#include <tinynn/nn/parameter_view.h>
#include <tinynn/nn/sequential.h>
#include <tinynn/training/callback.h>
#include <tinynn/training/callbacks/checkpoint_stream.h>
#include <tinynn/training/trainer.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tinynn {

// ------------------------------
// Full checkpoint format (v3)
// ------------------------------
//
// magic: 8 bytes = "TNYNSTT\0"
// u32 version = 3
// u32 type_size = sizeof(T)
//
// -- trainer --
// i32 next_epoch
// u64 global_step
//
// -- model params (id-based) --
// u64 num_params
// repeated:
//   u64 id
//   u8  kind
//   u32 rank
//   u64 dims[rank]
//   u64 count
//   raw T[count]
//
// -- model buffers (id-based) --
// u64 num_buffers
// repeated:
//   u64 id
//   u32 rank
//   u64 dims[rank]
//   u64 count
//   raw T[count]
//
// -- optimizer --
// string opt_type_tag
// T lr
// T weight_decay
// u64 opt_blob_size
// u8  opt_blob[opt_blob_size]
//
// -- callbacks --
// u32 num_entries
// repeated:
//   string tag
//   u64 blob_size
//   u8  blob[blob_size]
//
// Notes:
// - v2 checkpoints are still loadable.
// - params and buffers must each have stable, non-zero ids.
// - ids must be unique across BOTH params and buffers.
//

template <class T>
inline void save_full_checkpoint(const std::string& path, Trainer<T>& trainer,
                                 int next_epoch) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("save_full_checkpoint(v3): cannot open file: " +
                             path);
  }
  CheckpointWriter w(out);

  // magic
  const char magic[8] = {'T', 'N', 'Y', 'N', 'S', 'T', 'T', '\0'};
  w.write_bytes(magic, sizeof(magic), "magic");

  // version + type_size
  w.write_u32(3u);
  w.write_u32(static_cast<uint32_t>(sizeof(T)));

  // trainer state
  w.write_pod<int32_t>(static_cast<int32_t>(next_epoch), "i32 next_epoch");
  w.write_u64(static_cast<uint64_t>(trainer.global_step()));

  // -----------------
  // model params
  // -----------------
  std::vector<ParameterView<T>> params;
  trainer.model().collect_parameter_views(params);

  std::unordered_set<uint64_t> seen_ids;
  seen_ids.reserve(params.size() * 2 + 8);

  for (size_t i = 0; i < params.size(); ++i) {
    const auto& pv = params[i];
    const uint64_t id = static_cast<uint64_t>(pv.id);
    if (id == 0) {
      throw std::runtime_error(
          "save_full_checkpoint(v3): ParameterView.id must be non-zero "
          "(param index=" + std::to_string(i) + ")");
    }
    if (!seen_ids.insert(id).second) {
      throw std::runtime_error(
          "save_full_checkpoint(v3): duplicate ParameterView.id=" +
          std::to_string(id));
    }
  }

  w.write_u64(static_cast<uint64_t>(params.size()));

  for (size_t i = 0; i < params.size(); ++i) {
    const auto& pv = params[i];

    // id
    w.write_u64(static_cast<uint64_t>(pv.id));

    // kind
    w.write_u8(static_cast<uint8_t>(pv.kind));

    // shape rank/dims
    const uint32_t rank = static_cast<uint32_t>(pv.param.shape().rank());
    w.write_u32(rank);
    for (uint32_t d = 0; d < rank; ++d) {
      const uint64_t dim = static_cast<uint64_t>(
          pv.param.shape().dim_unchecked(static_cast<SizeType>(d)));
      w.write_u64(dim);
    }

    // count + bytes
    const uint64_t count = static_cast<uint64_t>(pv.param.size());
    w.write_u64(count);

    const T* src = pv.param.data();
    if (count > 0 && src == nullptr) {
      throw std::runtime_error(
          "save_full_checkpoint(v3): param data is null at param index=" +
          std::to_string(i) + " id=" +
          std::to_string(static_cast<uint64_t>(pv.id)));
    }
    w.write_bytes(src, static_cast<size_t>(sizeof(T) * count), "param bytes");
  }

  // -----------------
  // model buffers
  // -----------------
  std::vector<BufferView<T>> bufs;
  trainer.model().collect_buffer_views(bufs);

  for (size_t i = 0; i < bufs.size(); ++i) {
    const auto& bv = bufs[i];
    const uint64_t id = static_cast<uint64_t>(bv.id);
    if (id == 0) {
      throw std::runtime_error(
          "save_full_checkpoint(v3): BufferView.id must be non-zero "
          "(buffer index=" + std::to_string(i) + ")");
    }
    if (!seen_ids.insert(id).second) {
      throw std::runtime_error(
          "save_full_checkpoint(v3): duplicate id across params/buffers id=" +
          std::to_string(id));
    }
  }

  w.write_u64(static_cast<uint64_t>(bufs.size()));

  for (size_t i = 0; i < bufs.size(); ++i) {
    const auto& bv = bufs[i];

    // id
    w.write_u64(static_cast<uint64_t>(bv.id));

    // shape rank/dims
    const uint32_t rank = static_cast<uint32_t>(bv.buf.shape().rank());
    w.write_u32(rank);
    for (uint32_t d = 0; d < rank; ++d) {
      const uint64_t dim = static_cast<uint64_t>(
          bv.buf.shape().dim_unchecked(static_cast<SizeType>(d)));
      w.write_u64(dim);
    }

    // count + bytes
    const uint64_t count = static_cast<uint64_t>(bv.buf.size());
    w.write_u64(count);

    const T* src = bv.buf.data();
    if (count > 0 && src == nullptr) {
      throw std::runtime_error(
          "save_full_checkpoint(v3): buffer data is null at buffer index=" +
          std::to_string(i) + " id=" +
          std::to_string(static_cast<uint64_t>(bv.id)));
    }
    w.write_bytes(src, static_cast<size_t>(sizeof(T) * count), "buffer bytes");
  }

  // -----------------
  // optimizer
  // -----------------
  auto& opt = trainer.optimizer();
  w.write_string(std::string(opt.checkpoint_tag()));
  w.write_pod<T>(opt.learning_rate(), "lr");
  w.write_pod<T>(opt.weight_decay(), "weight_decay");

  std::ostringstream oss(std::ios::binary);
  CheckpointWriter ow(oss);
  opt.save_state(ow);
  const std::string blob = oss.str();

  w.write_u64(static_cast<uint64_t>(blob.size()));
  w.write_bytes(blob.data(), blob.size(), "opt blob");

  // -----------------
  // callbacks
  // -----------------
  std::vector<std::pair<std::string, std::string>> entries;
  trainer.for_each_callback([&](TrainerCallback<T>& cb) {
    const char* tag = cb.checkpoint_tag();
    if (!tag) return;

    std::ostringstream css(std::ios::binary);
    CheckpointWriter cw(css);
    cb.save_state(cw);

    entries.emplace_back(std::string(tag), css.str());
  });

  w.write_u32(static_cast<uint32_t>(entries.size()));
  for (auto& e : entries) {
    w.write_string(e.first);
    w.write_u64(static_cast<uint64_t>(e.second.size()));
    w.write_bytes(e.second.data(), e.second.size(), "cb blob");
  }
}

template <class T>
inline void load_full_checkpoint(const std::string& path, Trainer<T>& trainer,
                                 int* out_next_epoch) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("load_full_checkpoint: cannot open file: " + path);
  }
  CheckpointReader r(in);

  // magic
  char magic[8]{};
  r.read_bytes(magic, sizeof(magic), "magic");
  const char expected[8] = {'T', 'N', 'Y', 'N', 'S', 'T', 'T', '\0'};
  if (std::memcmp(magic, expected, 8) != 0) {
    throw std::runtime_error("load_full_checkpoint: magic mismatch");
  }

  const uint32_t version = r.read_u32();
  if (version != 2u && version != 3u) {
    throw std::runtime_error(
        "load_full_checkpoint: unsupported version=" +
        std::to_string(version));
  }

  const uint32_t type_size = r.read_u32();
  if (type_size != static_cast<uint32_t>(sizeof(T))) {
    throw std::runtime_error("load_full_checkpoint: type_size mismatch");
  }

  // trainer
  const int32_t next_epoch_i32 = r.read_pod<int32_t>("i32 next_epoch");
  const uint64_t global_step = r.read_u64();

  if (out_next_epoch) {
    *out_next_epoch = static_cast<int>(next_epoch_i32);
  }
  trainer.set_global_step(static_cast<SizeType>(global_step));

  // -----------------
  // model params (id-based)
  // -----------------
  const uint64_t file_num_params = r.read_u64();

  std::vector<ParameterView<T>> params;
  trainer.model().collect_parameter_views(params);

  if (file_num_params != static_cast<uint64_t>(params.size())) {
    throw std::runtime_error(
        "load_full_checkpoint: num_params mismatch (file=" +
        std::to_string(file_num_params) +
        ", current=" + std::to_string(params.size()) + ")");
  }

  std::unordered_map<uint64_t, ParameterView<T>*> by_id;
  by_id.reserve(params.size() * 2 + 1);

  for (size_t i = 0; i < params.size(); ++i) {
    auto& pv = params[i];
    const uint64_t id = static_cast<uint64_t>(pv.id);
    if (id == 0) {
      throw std::runtime_error(
          "load_full_checkpoint: current ParameterView.id must be non-zero "
          "(index=" + std::to_string(i) + ")");
    }
    auto [it, inserted] = by_id.emplace(id, &pv);
    if (!inserted) {
      throw std::runtime_error(
          "load_full_checkpoint: duplicate current ParameterView.id=" +
          std::to_string(id));
    }
  }

  for (uint64_t k = 0; k < file_num_params; ++k) {
    const uint64_t id = r.read_u64();

    auto it = by_id.find(id);
    if (it == by_id.end()) {
      throw std::runtime_error("load_full_checkpoint: unknown param id=" +
                               std::to_string(id));
    }

    ParameterView<T>& pv = *it->second;

    // kind
    const uint8_t file_kind_u8 = r.read_u8();
    const uint8_t expected_kind_u8 = static_cast<uint8_t>(pv.kind);
    if (file_kind_u8 != expected_kind_u8) {
      throw std::runtime_error(
          "load_full_checkpoint: ParamKind mismatch at id=" +
          std::to_string(id));
    }

    // shape
    const uint32_t rank = r.read_u32();
    if (rank != static_cast<uint32_t>(pv.param.shape().rank())) {
      throw std::runtime_error(
          "load_full_checkpoint: rank mismatch at id=" +
          std::to_string(id));
    }
    for (uint32_t d = 0; d < rank; ++d) {
      const uint64_t dim = r.read_u64();
      const uint64_t expected_dim = static_cast<uint64_t>(
          pv.param.shape().dim_unchecked(static_cast<SizeType>(d)));
      if (dim != expected_dim) {
        throw std::runtime_error(
            "load_full_checkpoint: dim mismatch at id=" +
            std::to_string(id));
      }
    }

    // count + bytes
    const uint64_t count = static_cast<uint64_t>(pv.param.size());
    const uint64_t file_count = r.read_u64();
    if (file_count != count) {
      throw std::runtime_error(
          "load_full_checkpoint: count mismatch at id=" +
          std::to_string(id));
    }

    T* dst = pv.param.data();
    if (file_count > 0 && dst == nullptr) {
      throw std::runtime_error(
          "load_full_checkpoint: param data is null at id=" +
          std::to_string(id));
    }
    r.read_bytes(dst, static_cast<size_t>(sizeof(T) * file_count),
                 "param bytes");
  }

  // -----------------
  // model buffers (v3 only)
  // -----------------
  if (version == 3u) {
    const uint64_t file_num_bufs = r.read_u64();

    std::vector<BufferView<T>> bufs;
    trainer.model().collect_buffer_views(bufs);

    if (file_num_bufs != static_cast<uint64_t>(bufs.size())) {
      throw std::runtime_error(
          "load_full_checkpoint(v3): num_buffers mismatch (file=" +
          std::to_string(file_num_bufs) +
          ", current=" + std::to_string(bufs.size()) + ")");
    }

    std::unordered_map<uint64_t, BufferView<T>*> buf_by_id;
    buf_by_id.reserve(bufs.size() * 2 + 1);

    for (size_t i = 0; i < bufs.size(); ++i) {
      auto& bv = bufs[i];
      const uint64_t id = static_cast<uint64_t>(bv.id);
      if (id == 0) {
        throw std::runtime_error(
            "load_full_checkpoint(v3): current BufferView.id must be non-zero "
            "(index=" + std::to_string(i) + ")");
      }
      auto [it, inserted] = buf_by_id.emplace(id, &bv);
      if (!inserted) {
        throw std::runtime_error(
            "load_full_checkpoint(v3): duplicate current BufferView.id=" +
            std::to_string(id));
      }
      if (by_id.find(id) != by_id.end()) {
        throw std::runtime_error(
            "load_full_checkpoint(v3): id collision between current params and "
            "buffers id=" + std::to_string(id));
      }
    }

    for (uint64_t k = 0; k < file_num_bufs; ++k) {
      const uint64_t id = r.read_u64();

      auto it = buf_by_id.find(id);
      if (it == buf_by_id.end()) {
        throw std::runtime_error(
            "load_full_checkpoint(v3): unknown buffer id=" +
            std::to_string(id));
      }

      BufferView<T>& bv = *it->second;

      // shape
      const uint32_t rank = r.read_u32();
      if (rank != static_cast<uint32_t>(bv.buf.shape().rank())) {
        throw std::runtime_error(
            "load_full_checkpoint(v3): rank mismatch at buffer id=" +
            std::to_string(id));
      }
      for (uint32_t d = 0; d < rank; ++d) {
        const uint64_t dim = r.read_u64();
        const uint64_t expected_dim = static_cast<uint64_t>(
            bv.buf.shape().dim_unchecked(static_cast<SizeType>(d)));
        if (dim != expected_dim) {
          throw std::runtime_error(
              "load_full_checkpoint(v3): dim mismatch at buffer id=" +
              std::to_string(id));
        }
      }

      // count + bytes
      const uint64_t count = static_cast<uint64_t>(bv.buf.size());
      const uint64_t file_count = r.read_u64();
      if (file_count != count) {
        throw std::runtime_error(
            "load_full_checkpoint(v3): count mismatch at buffer id=" +
            std::to_string(id));
      }

      T* dst = bv.buf.data();
      if (file_count > 0 && dst == nullptr) {
        throw std::runtime_error(
            "load_full_checkpoint(v3): buffer data is null at id=" +
            std::to_string(id));
      }
      r.read_bytes(dst, static_cast<size_t>(sizeof(T) * file_count),
                   "buffer bytes");
    }
  }

  // -----------------
  // optimizer
  // -----------------
  const std::string opt_tag = r.read_string();
  const T lr = r.read_pod<T>("lr");
  const T wd = r.read_pod<T>("weight_decay");

  const uint64_t opt_blob_size = r.read_u64();
  std::string opt_blob;
  opt_blob.resize(static_cast<size_t>(opt_blob_size));
  if (opt_blob_size) {
    r.read_bytes(opt_blob.data(), static_cast<size_t>(opt_blob_size),
                 "opt blob");
  }

  auto& opt = trainer.optimizer();
  if (opt_tag != std::string(opt.checkpoint_tag())) {
    throw std::runtime_error(
        "load_full_checkpoint: optimizer type mismatch (file=" + opt_tag +
        ", current=" + std::string(opt.checkpoint_tag()) + ")");
  }
  opt.set_learning_rate(lr);
  opt.set_weight_decay(wd);

  {
    std::istringstream iss(opt_blob, std::ios::binary);
    CheckpointReader orr(iss);
    opt.load_state(orr);
  }

  // -----------------
  // callbacks
  // -----------------
  const uint32_t num_entries = r.read_u32();

  std::unordered_map<std::string, std::string> blobs;
  blobs.reserve(num_entries);

  for (uint32_t i = 0; i < num_entries; ++i) {
    const std::string tag = r.read_string();
    const uint64_t sz = r.read_u64();

    std::string b;
    b.resize(static_cast<size_t>(sz));
    if (sz) {
      r.read_bytes(b.data(), static_cast<size_t>(sz), "cb blob");
    }
    blobs.emplace(tag, std::move(b));
  }

  trainer.for_each_callback([&](TrainerCallback<T>& cb) {
    const char* tag = cb.checkpoint_tag();
    if (!tag) return;

    auto it = blobs.find(tag);
    if (it == blobs.end()) return;  // missing -> keep defaults

    std::istringstream iss(it->second, std::ios::binary);
    CheckpointReader cr(iss);
    cb.load_state(cr);
  });

  // trailing bytes check
  {
    char extra = 0;
    in.read(&extra, 1);
    if (in.gcount() == 1) {
      throw std::runtime_error(
          "load_full_checkpoint: trailing bytes detected");
    }
    if (!in.eof()) {
      throw std::runtime_error(
          "load_full_checkpoint: failed to reach EOF cleanly");
    }
    in.clear();
  }
}

}  // namespace tinynn
