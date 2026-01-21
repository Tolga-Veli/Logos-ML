#include <new>
#include <stdexcept>

#include "AlignedAlloc.hpp"
#include "Buffer.hpp"

namespace Logos::Memory {
Buffer::Buffer()
    : m_Data(nullptr), m_Bytes(0), m_Alignment(DEFAULT_ALIGNMENT) {}

Buffer::Buffer(std::size_t size, std::size_t alignment)
    : m_Data(nullptr), m_Bytes(0), m_Alignment(alignment) {
  reset(size, alignment);
}

Buffer::~Buffer() { aligned_free(m_Data); }

Buffer::Buffer(Buffer &&other) noexcept
    : m_Data(other.m_Data), m_Bytes(other.m_Bytes),
      m_Alignment(other.m_Alignment) {
  other.m_Data = nullptr;
  other.m_Bytes = 0;
  other.m_Alignment = DEFAULT_ALIGNMENT;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (this == &other)
    return *this;

  aligned_free(m_Data);
  m_Data = other.m_Data;
  m_Bytes = other.m_Bytes;
  m_Alignment = other.m_Alignment;

  other.m_Data = nullptr;
  other.m_Bytes = 0;
  other.m_Alignment = DEFAULT_ALIGNMENT;
  return *this;
}

void Buffer::reset(std::size_t size, std::size_t alignment) {
  if (!IsPow2(alignment))
    throw std::logic_error("Buffer alignment must be a power of two");
  if (alignment < alignof(void *))
    throw std::logic_error("Buffer alignment is too small");

  aligned_free(m_Data);
  m_Data = nullptr;
  m_Bytes = 0;

  if (size == 0) {
    m_Alignment = alignment;
    return;
  }

  m_Data = aligned_malloc(size, alignment);
  if (!m_Data)
    throw std::bad_alloc();

  m_Bytes = size;
  m_Alignment = alignment;
}
} // namespace Logos::Memory
