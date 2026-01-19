#pragma once

#include <cstdint>
#include <cstdlib>

namespace logos {
enum class Memory : std::uint64_t {
  B = 1,
  KiB = 1 << 10,
  MiB = 1 << 20,
  GiB = 1 << 30,
  TiB = 1 << 40
};

class MemoryArena {
public:
  MemoryArena() = delete;
  MemoryArena(Memory capacity) {
    std::uint64_t mem = static_cast<std::uint64_t>(capacity);
    m_Arena = (ArenaInfo *)malloc(mem);

    m_Arena->cap = mem;
    m_Arena->pos = sizeof(m_Arena);
  }

  ~MemoryArena() { free(m_Arena); }

private:
  struct ArenaInfo {
    std::uint64_t pos, cap;
  };
  ArenaInfo *m_Arena;
};

} // namespace logos