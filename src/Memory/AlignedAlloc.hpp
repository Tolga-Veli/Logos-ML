#pragma once

#include <cstddef>

namespace Logos::Memory {

#if defined(_WIN32)

#include <malloc.h>
inline void *aligned_malloc(std::size_t size, std::size_t alignment) {
  return _aligned_malloc(size, alignment);
}

inline void aligned_free(void *ptr) { _aligned_free(ptr); }

#else

#include <cstdlib>
inline void *aligned_malloc(std::size_t size, std::size_t alignment) {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size))
    return nullptr;
  return ptr;
}

inline void aligned_free(void *ptr) { free(ptr); }

#endif
} // namespace Logos::Memory
