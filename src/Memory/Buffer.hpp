#pragma once

#include <cstddef>
#include <cstring>
#include <new>
#include <stdexcept>

#include "MemoryUtility.hpp"

namespace Logos::Memory {
class Buffer {
public:
  Buffer() noexcept = default;
  explicit Buffer(std::size_t size, std::size_t alignment = DEFAULT_ALIGNMENT);
  ~Buffer();

  Buffer(const Buffer &other) = delete;
  Buffer &operator=(const Buffer &other) = delete;

  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

  void allocate(std::size_t size, std::size_t alignment = DEFAULT_ALIGNMENT);
  void release() noexcept;

  explicit operator bool() const noexcept { return m_Data != nullptr; }

  std::byte *data() noexcept { return m_Data; }
  const std::byte *data() const noexcept { return m_Data; }

  template <class T> T *as() noexcept {
    if (!m_Data)
      return nullptr;
    return reinterpret_cast<T *>(m_Data);
  }

  template <class T> const T *as() const noexcept {
    if (!m_Data)
      return nullptr;
    return reinterpret_cast<const T *>(m_Data);
  }

  void clear() noexcept { std::memset(m_Data, 0, m_Bytes); }

  std::size_t size() const noexcept { return m_Bytes; }
  std::size_t alignment() const noexcept { return m_Alignment; }

private:
  std::byte *m_Data = nullptr;
  std::size_t m_Bytes = 0, m_Alignment = DEFAULT_ALIGNMENT;
};
} // namespace Logos::Memory
