#pragma once

#include <cstddef>
#include <cstring>

#include "MemoryUtility.hpp"
/*
namespace Logos::Memory {
class SharedBuffer {
public:
  SharedBuffer() = delete;
  explicit SharedBuffer(std::size_t size,
                        std::size_t alignment = DEFAULT_ALIGNMENT);
  ~SharedBuffer();

  SharedBuffer(const SharedBuffer &other) = delete;
  SharedBuffer &operator=(const SharedBuffer &other) = delete;

  SharedBuffer(SharedBuffer &&other) noexcept;
  SharedBuffer &operator=(SharedBuffer &&other) noexcept;

  explicit operator bool() const noexcept { return m_Data != nullptr; }

  std::byte *data() noexcept { return m_Data; }
  const std::byte *data() const noexcept { return m_Data; }

  template <class T> T *as() {
    if (!m_Data)
      return nullptr;
    return reinterpret_cast<T *>(m_Data);
  }

  template <class T> const T *as() const {
    if (!m_Data)
      return nullptr;
    return reinterpret_cast<const T *>(m_Data);
  }

  void clear() { std::memset(m_Data, 0, m_Bytes); }
  std::size_t size() const noexcept { return m_Bytes; }
  std::size_t alignment() const noexcept { return m_Alignment; }

private:
  std::byte *m_Data;
  std::size_t m_Bytes, m_Alignment;

  void allocate(std::size_t size, std::size_t alignment = DEFAULT_ALIGNMENT);
  void release() noexcept;
};
} */// namespace Logos::Memory
