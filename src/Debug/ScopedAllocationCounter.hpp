#pragma once

#include "Core/Logging.hpp"
#include "Memory/MemoryStats.hpp"

namespace ml::debug {
class ScopedAllocationCounter {
public:
  explicit ScopedAllocationCounter(bool log = false) : m_ShouldLog(log) {
    if (log)
      m_StartMemoryStats = ml::memory::get_stats();
  }

  ~ScopedAllocationCounter() {
    if (m_ShouldLog) {
      const auto &stats = ml::memory::get_stats();

      LOG_INFO("Allocations: {}, Deallocations: {} Current Bytes Allocated: "
               "{},  Total Bytes Allocated: {}, Total Bytes Deallocated: {}",
               stats.allocation_count() - m_StartMemoryStats.allocation_count(),
               stats.deallocation_count() -
                   m_StartMemoryStats.deallocation_count(),
               stats.current_bytes() - m_StartMemoryStats.current_bytes(),
               stats.total_bytes_allocated() -
                   m_StartMemoryStats.total_bytes_allocated(),
               stats.total_bytes_deallocated() -
                   m_StartMemoryStats.total_bytes_deallocated());
    }
  }

private:
  memory::MemoryStats m_StartMemoryStats{};
  bool m_ShouldLog;
};
} // namespace ml::debug
