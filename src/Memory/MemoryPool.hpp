#pragma once

#include "Buffer.hpp"

#include <stdexcept>

namespace Logos::Memory {

class Arena {
public:
  Arena() = delete;
  explicit Arena(std::size_t size, std::size_t alignment = DEFAULT_ALIGNMENT)
      : m_Buffer(size, alignment),
        m_Base(reinterpret_cast<std::byte *>(m_Buffer.data())),
        m_Capacity(size), m_Offset(0), m_DefaultAlignment(alignment) {}
  ~Arena() = default;

  Arena(const Arena &) = delete;
  Arena &operator=(const Arena &) = delete;

  void reset() noexcept { m_Offset = 0; }

  std::size_t used() const noexcept { return m_Offset; }
  std::size_t capacity() const noexcept { return m_Capacity; }
  std::size_t remaining() const noexcept { return m_Capacity - m_Offset; }

  template <class T>
  T *Allocate(std::size_t count = 1, std::size_t alignment = alignof(T)) {
    if (!IsPow2(alignment))
      throw std::logic_error("Arena alignment must be a power of two");

    std::size_t start = AlignUp(m_Offset, alignment);
    std::size_t end = start + sizeof(T) * count;

    if (end > m_Capacity)
      throw std::bad_alloc();

    m_Offset = end;
    return reinterpret_cast<T *>(m_Base + start);
  }

private:
  Buffer m_Buffer;
  std::byte *m_Base;
  std::size_t m_Capacity, m_Offset, m_DefaultAlignment;
};
} // namespace Logos::Memory
