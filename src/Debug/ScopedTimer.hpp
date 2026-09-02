#pragma once

#include "Core/Logging.hpp"

#include <chrono>

namespace ml::debug {
class ScopedTimer {
public:
  explicit ScopedTimer(bool log = false) : m_ShouldLog(log) {
    if (log)
      m_Start = Clock::now();
  }

  ~ScopedTimer() {
    if (m_ShouldLog)
      LOG_INFO("Duration: {}", CalculateDuration());
  }

  void Start() { m_Start = Clock::now(); }

  [[nodiscard]] std::chrono::milliseconds GetDuration() const {
    return CalculateDuration();
  }

private:
  using Clock = std::chrono::steady_clock;

  Clock::time_point m_Start;
  bool m_ShouldLog;

  std::chrono::milliseconds CalculateDuration() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                                 m_Start);
  }
};
} // namespace ml::debug
