#pragma once

#include "Buffer.hpp"

#include <stdexcept>

namespace Logos::Memory {
// Arena / Bump-Allocator that behaves similarly to a stack
// allocate a chunk of memory and keep an offset variable that you push for each
// allocation I'm using a Buffer to do the allocations Arena doesn't call
// destructors. Objects are destroyed when the Arena goes out of scope
class Arena {
public:
  Arena() = delete;
  explicit Arena(std::size_t size, std::size_t alignment = DEFAULT_ALIGNMENT)
      : m_Buffer(size, alignment), m_Offset(0) {
    if (!IsPow2(alignment))
      throw std::logic_error("Arena alignment must be a power of two");
  }
  ~Arena() = default;

  Arena(const Arena &) = delete;
  Arena &operator=(const Arena &) = delete;

  Arena(Arena &&other) noexcept = delete; /*
      : m_Buffer(std::move(other.m_Buffer)), m_Offset(other.m_Offset) {
    other.m_Offset = 0;
  }*/
  Arena &operator=(Arena &&other) noexcept = delete;
  /*{
    if (this == &other)
      return *this;

    m_Buffer.release();
    m_Buffer = std::move(other.m_Buffer);
    m_Offset = other.m_Offset;
    other.m_Offset = 0;
    return *this;
  }*/

  std::size_t offset() const noexcept { return m_Offset; }
  std::size_t capacity() const noexcept { return m_Buffer.size(); }
  std::size_t remaining() const noexcept { return m_Buffer.size() - m_Offset; }

  void reset() noexcept { m_Offset = 0; }

  void rewind(std::size_t offset) noexcept {
    if (offset <= m_Offset)
      m_Offset = offset;
  }

  template <class T>
  T *allocate(std::size_t count = 1, std::size_t alignment = alignof(T)) {
    if (!count)
      return nullptr;

    if (!IsPow2(alignment))
      throw std::logic_error("Arena alignment must be a power of two");

    const std::size_t start = AlignUp(m_Offset, alignment);
    const std::uint64_t bytes = (std::uint64_t)sizeof(T) * count;

    if (start > capacity() || bytes > capacity() - start)
      throw std::runtime_error("Arena::allocate");

    m_Offset = start + bytes;
    return reinterpret_cast<T *>(base() + start);
  }

  // Allocates and constructs a T object on the arena
  template <class T, class... Args> T *create(Args &&...args) {
    const auto curr_off = m_Offset;
    T *ptr = allocate<T>();
    try {
      ::new (static_cast<void *>(ptr)) T(std::forward<Args>(args)...);
    } catch (...) {
      m_Offset = curr_off;
      throw;
    }
    return ptr;
  }

private:
  Buffer m_Buffer;
  std::size_t m_Offset;

  std::byte *base() noexcept { return m_Buffer.data(); }
  const std::byte *base() const noexcept { return m_Buffer.data(); }
};
} // namespace Logos::Memory
