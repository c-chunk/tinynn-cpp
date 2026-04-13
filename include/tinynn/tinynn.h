#pragma once

#include <cstddef>
#include <cstdint>

// Version: 0.1.0 (semantic versioning)
#define TINYNN_VERSION_MAJOR 0
#define TINYNN_VERSION_MINOR 1
#define TINYNN_VERSION_PATCH 0

// Compiler helpers
#if defined(_MSC_VER)
#define TINYNN_MSVC 1
#else
#define TINYNN_MSVC 0
#endif

namespace tinynn {

// Library-wide common types
using SizeType = std::size_t;

// Internal implementation details should live here.
namespace detail {}  // namespace detail

}  // namespace tinynn
