#pragma once

#include "Arena.hpp"

namespace Logos::Memory {
// ScratchArena is used for temporary memory and must be outlived by the main
// Arena it references
class ScratchArena {
public:
  explicit ScratchArena(Logos::Memory::Arena &arena)
      : m_Arena(arena), m_BeginOffset(m_Arena.offset()) {}

  ScratchArena(const ScratchArena &) = delete;
  ScratchArena &operator=(const ScratchArena &) = delete;

  ~ScratchArena() noexcept { m_Arena.rewind(m_BeginOffset); }

  template <class T>
  T *allocate(std::size_t count = 1, std::size_t alignment = alignof(T)) {
    return m_Arena.allocate<T>(count, alignment);
  }

  std::size_t offset() const { return m_Arena.offset(); }
  void rewind(std::size_t offset) { m_Arena.rewind(offset); }

private:
  Logos::Memory::Arena &m_Arena;
  std::size_t m_BeginOffset;
};

} // namespace Logos::Memory
