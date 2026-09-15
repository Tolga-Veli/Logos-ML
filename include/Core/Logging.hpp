#pragma once

#include <format>
#include <print>
#include <source_location>
#include <string_view>
#include <utility>

namespace ml::core {
enum class LogLevel : std::uint8_t { Debug, Info, Warn, Error, Fatal };

namespace detail {
[[nodiscard]] constexpr std::string_view to_string(LogLevel level) noexcept {
  using enum LogLevel;
  switch (level) {
  case Debug:
    return "DEBUG";
  case Info:
    return "INFO";
  case Warn:
    return "WARN";
  case Error:
    return "ERROR";
  case Fatal:
    return "FATAL";
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

class Logger final {
public:
  [[nodiscard]] static Logger &GetInstance() noexcept {
    static Logger instance;
    return instance;
  }

  void SetMinLevel(LogLevel level) noexcept { m_minLevel = level; }
  [[nodiscard]] LogLevel MinLevel() const noexcept { return m_minLevel; }

  void SetColorEnabled(bool enabled) noexcept { m_colorEnabled = enabled; }
  [[nodiscard]] bool ColorEnabled() const noexcept { return m_colorEnabled; }

  template <typename... Args>
  void Log(LogLevel level, std::source_location loc,
           std::format_string<Args...> format, Args &&...args) {
    if (level < m_minLevel)
      return;

    FILE *stream = StreamFor(level);
    PrintPrefix(stream, level);

    std::print(stream, format, std::forward<Args>(args)...);
    std::println(stream, " ({}:{})", loc.file_name(), loc.line());
    std::fflush(stream);
  }

  template <typename... Args>
  void Log(LogLevel level, std::format_string<Args...> format, Args &&...args) {
    if (level < m_minLevel)
      return;

    FILE *stream = StreamFor(level);
    PrintPrefix(stream, level);
    std::println(stream, format, std::forward<Args>(args)...);
    std::fflush(stream);
  }

private:
  static constexpr std::string_view k_ColorReset = "\x1b[0m";

  Logger() = default;
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  [[nodiscard]] static FILE *StreamFor(LogLevel level) noexcept {
    return level >= LogLevel::Warn ? stderr : stdout;
  }

  void PrintPrefix(FILE *stream, LogLevel level) const {
    if (m_colorEnabled) {
      std::print(stream, "{}[{}]{} ", to_ansi_color(level), to_string(level),
                 k_ColorReset);
      return;
    }

    std::print(stream, "[{}] ", to_string(level));
  }

  LogLevel m_minLevel = LogLevel::Debug;
  bool m_colorEnabled = true;
};
} // namespace detail
} // namespace ml::core

// std::source_location::current() is captured at the macro call site, so
// every log line points at the code that actually logged it, not at some
// wrapper function three layers down.
#define LOG(level, ...)                                                        \
  ::ml::core::detail::Logger::GetInstance().Log(level, __VA_ARGS__)

#define LOG_SOURCE_LOC(level, ...)                                             \
  ::ml::core::detail::Logger::GetInstance().Log(                               \
      level, std::source_location::current(), __VA_ARGS__)

#define LOG_DEBUG(...) LOG_SOURCE_LOC(::ml::core::LogLevel::Debug, __VA_ARGS__)
#define LOG_INFO(...) LOG(::ml::core::LogLevel::Info, __VA_ARGS__)
#define LOG_WARN(...) LOG_SOURCE_LOC(::ml::core::LogLevel::Warn, __VA_ARGS__)
#define LOG_ERROR(...) LOG_SOURCE_LOC(::ml::core::LogLevel::Error, __VA_ARGS__)
