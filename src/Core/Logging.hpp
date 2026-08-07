#pragma once

#include <cstdio>
#include <format>
#include <mutex>
#include <source_location>
#include <string_view>
#include <utility>

namespace ml::core {
enum class LogLevel : std::uint8_t {
  Trace = 0,
  Debug,
  Info,
  Warn,
  Error,
  Fatal
};

namespace detail {
[[nodiscard]] constexpr std::string_view to_string(LogLevel level) noexcept {
  using enum LogLevel;
  switch (level) {
  case Trace:
    return "TRACE";
  case Debug:
    return "DEBUG";
  case Info:
    return "INFO";
  case Warn:
    return "WARN";
  case Error:
    return "ERROR";
  case Fatal:
  default:
    return "Unknown";
  }
}

// ANSI escape codes for terminal output. Harmless if the target stream
// isn't a color-capable terminal; Logger::SetColorEnabled can disable
// emitting these entirely (e.g. redirecting to a file).
//
// Note: these are terminal codes, not usable directly.
[[nodiscard]] constexpr std::string_view
to_ansi_color(LogLevel level) noexcept {
  using enum LogLevel;
  switch (level) {
  case Trace:
    return "\x1b[90m";
  case Debug:
    return "\x1b[36m";
  case Info:
    return "\x1b[32m";
  case Warn:
    return "\x1b[33m";
  case Error:
    return "\x1b[31m";
  case Fatal:
    return "\x1b[41;97m";
  default:
    return "";
  }
}

inline constexpr std::string_view k_ColorReset = "\x1b[0m";

class Logger {
public:
  static Logger &Instance() noexcept {
    static Logger instance;
    return instance;
  }

  void SetMinLevel(LogLevel level) noexcept { m_minLevel = level; }
  [[nodiscard]] LogLevel MinLevel() const noexcept { return m_minLevel; }

  void SetColorEnabled(bool enabled) noexcept { m_colorEnabled = enabled; }

  template <typename... Args>
  void Log(LogLevel level, std::source_location loc,
           std::format_string<Args...> fmt, Args &&...args) {
    if (level < m_minLevel)
      return;

    std::string message = std::format(fmt, std::forward<Args>(args)...);

    std::scoped_lock lock(m_mutex);
    FILE *stream = (level >= LogLevel::Warn) ? stderr : stdout;

    if (m_colorEnabled) {
      std::fprintf(stream, "%s[%s]%s %s (%s:%d)\n", to_ansi_color(level).data(),
                   to_string(level).data(), k_ColorReset.data(),
                   message.c_str(), loc.file_name(),
                   static_cast<int>(loc.line()));
    } else {
      std::fprintf(stream, "[%s] %s (%s:%d)\n", to_string(level).data(),
                   message.c_str(), loc.file_name(),
                   static_cast<int>(loc.line()));
    }
    std::fflush(stream);
  }

private:
  Logger() = default;

  LogLevel m_minLevel = LogLevel::Trace;
  bool m_colorEnabled = true;
  std::mutex m_mutex;
};
} // namespace detail
} // namespace ml::core

// std::source_location::current() is captured at the macro call site, so
// every log line points at the code that actually logged it, not at some
// wrapper function three layers down.
#define LOG(level, ...)                                                        \
  ::ml::core::Logger::Instance().Log(level, std::source_location::current(),   \
                                     __VA_ARGS__)

#define LOG_TRACE(...) LOG(::ml::core::LogLevel::Trace, __VA_ARGS__)
#define LOG_DEBUG(...) LOG(::ml::core::LogLevel::Debug, __VA_ARGS__)
#define LOG_INFO(...) LOG(::ml::core::LogLevel::Info, __VA_ARGS__)
#define LOG_WARN(...) LOG(::ml::core::LogLevel::Warn, __VA_ARGS__)
#define LOG_ERROR(...) LOG(::ml::core::LogLevel::Error, __VA_ARGS__)
#define LOG_FATAL(...) LOG(::ml::core::LogLevel::Fatal, __VA_ARGS__)
