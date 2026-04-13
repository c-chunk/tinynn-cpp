#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace tinynn {

class CheckpointWriter {
 public:
  explicit CheckpointWriter(std::ostream& os) : os_(os) {}

  void write_u8(uint8_t v) { write_pod(v, "u8"); }
  void write_u32(uint32_t v) { write_pod(v, "u32"); }
  void write_u64(uint64_t v) { write_pod(v, "u64"); }

  template <class T>
  void write_pod(const T& v, const char* what) {
    os_.write(reinterpret_cast<const char*>(&v), sizeof(T));
    if (!os_) {
      throw std::runtime_error(
          std::string("CheckpointWriter: write ") + what + " failed");
    }
  }

  void write_bytes(const void* data, size_t n, const char* what = "bytes") {
    if (n == 0) {
      return;
    }
    os_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (!os_) {
      throw std::runtime_error(
          std::string("CheckpointWriter: write ") + what + " failed");
    }
  }

  void write_string(const std::string& s) {
    if (s.size() > 0xFFFFFFFFu) {
      throw std::runtime_error("CheckpointWriter: string too long");
    }
    write_u32(static_cast<uint32_t>(s.size()));
    write_bytes(s.data(), s.size(), "string");
  }

 private:
  std::ostream& os_;
};

class CheckpointReader {
 public:
  explicit CheckpointReader(std::istream& is) : is_(is) {}

  uint8_t read_u8() { return read_pod<uint8_t>("u8"); }
  uint32_t read_u32() { return read_pod<uint32_t>("u32"); }
  uint64_t read_u64() { return read_pod<uint64_t>("u64"); }

  template <class T>
  T read_pod(const char* what) {
    T v{};
    is_.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!is_) {
      throw std::runtime_error(
          std::string("CheckpointReader: read ") + what + " failed");
    }
    return v;
  }

  void read_bytes(void* dst, size_t n, const char* what = "bytes") {
    if (n == 0) {
      return;
    }
    is_.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (!is_) {
      throw std::runtime_error(
          std::string("CheckpointReader: read ") + what + " failed");
    }
  }

  std::string read_string() {
    const uint32_t n = read_u32();
    std::string s;
    s.resize(n);
    if (n) {
      read_bytes(s.data(), n, "string");
    }
    return s;
  }

 private:
  std::istream& is_;
};

}  // namespace tinynn
