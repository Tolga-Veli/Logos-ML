#pragma once

#include <cstddef>
#include <cstring>

#include "MemoryUtility.hpp"

namespace Logos::Memory {
class Buffer {
public:
  Buffer();
  explicit Buffer(std::size_t size, std::size_t alignment = DEFAULT_ALIGNMENT);
  ~Buffer();

  Buffer(const Buffer &other) = delete;
  Buffer &operator=(const Buffer &other) = delete;

  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

  void reset(std::size_t size, std::size_t alignment = DEFAULT_ALIGNMENT);

  void fill_zeroes() { std::memset(m_Data, 0, m_Bytes); }
  void *data() noexcept { return m_Data; }
  const void *data() const noexcept { return m_Data; }

  std::size_t size_bytes() const noexcept { return m_Bytes; }
  std::size_t alignment() const noexcept { return m_Alignment; }

private:
  void *m_Data;
  std::size_t m_Bytes, m_Alignment;
};
} // namespace Logos::Memory
