#include "SharedBuffer.hpp"
#include "AlignedAlloc.hpp"
/*
namespace Logos::Memory {
SharedBuffer::SharedBuffer(std::size_t size, std::size_t alignment) {
  allocate(size, alignment);
}

SharedBuffer::~SharedBuffer() { release(); }

SharedBuffer::SharedBuffer(SharedBuffer &&other) noexcept
    : m_Data(other.m_Data), m_Bytes(other.m_Bytes),
      m_Alignment(other.m_Alignment) {
  other.m_Data = nullptr;
  other.m_Bytes = 0;
  other.m_Alignment = DEFAULT_ALIGNMENT;
}

SharedBuffer &SharedBuffer::operator=(SharedBuffer &&other) noexcept {
  if (this == &other)
    return *this;

  release();

  m_Data = other.m_Data;
  m_Bytes = other.m_Bytes;
  m_Alignment = other.m_Alignment;

  other.m_Data = nullptr;
  other.m_Bytes = 0;
  other.m_Alignment = DEFAULT_ALIGNMENT;

  return *this;
}

void SharedBuffer::allocate(std::size_t size, std::size_t alignment) {
  if (!IsPow2(alignment))
    throw std::logic_error("SharedBuffer alignment must be a power of two");

  release();

  if (size == 0) {
    m_Alignment = alignment;
    return;
  }

  m_Data = reinterpret_cast<std::byte *>(aligned_malloc(size, alignment));
  if (!m_Data)
    throw std::bad_alloc();

  m_Bytes = size;
  m_Alignment = alignment;
}

void SharedBuffer::release() noexcept {
  aligned_free(m_Data);
  m_Data = nullptr;
  m_Bytes = 0;
  m_Alignment = DEFAULT_ALIGNMENT;
}

} */// namespace Logos::Memory
