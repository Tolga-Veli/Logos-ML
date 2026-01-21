#pragma once

#include <cstdint>
#include <format>
#include <iostream>
#include <string_view>

namespace Logos::Core {

enum class LogLevel : uint8_t { Trace = 0, Info, Warning, Error, Fatal, None };

class Logger {
public:
  static void SetLevel(LogLevel level) { s_CurrentLevel = level; }
  static LogLevel GetLevel() { return s_CurrentLevel; }

  template <typename... Args>
  static void Log(LogLevel level, std::string_view prefix,
                  std::string_view color, std::format_string<Args...> fmt,
                  Args &&...args) {
    if (level < s_CurrentLevel)
      return;

    std::string message = std::format(fmt, std::forward<Args>(args)...);
    std::cout << color << "[" << prefix << "] " << message << "\033[0m"
              << std::endl;
  }

private:
  inline static LogLevel s_CurrentLevel = LogLevel::Trace;
};
} // namespace Logos::Core

#define LOGOS_COL_GRAY "\033[90m"
#define LOGOS_COL_CYAN "\033[36m"
#define LOGOS_COL_GREEN "\033[32m"
#define LOGOS_COL_YELLOW "\033[33m"
#define LOGOS_COL_RED "\033[31m"
#define LOGOS_COL_FATAL "\033[41m\033[37m"

#ifdef LOGOS_DEBUG
#define LOGOS_INFO(...)                                                        \
  ::Logos::Core::Logger::Log(::Logos::Core::LogLevel::Info, "INFO",            \
                             LOGOS_COL_GREEN, __VA_ARGS__)
#define LOGOS_WARN(...)                                                        \
  ::Logos::Core::Logger::Log(::Logos::Core::LogLevel::Warning, "WARN",         \
                             LOGOS_COL_CYAN, __VA_ARGS__)
#define LOGOS_ERROR(...)                                                       \
  ::Logos::Core::Logger::Log(::Logos::Core::LogLevel::Error, "ERROR",          \
                             LOGOS_COL_RED, __VA_ARGS__)
#define LOGOS_FATAL(...)                                                       \
  ::Logos::Core::Logger::Log(::Logos::Core::LogLevel::Fatal, "FATAL",          \
                             LOGOS_COL_FATAL, __VA_ARGS__),                    \
      throw std::runtime_error()

#else
#define LOGOS_INFO(...)
#define LOGOS_WARN(...)
#define LOGOS_ERROR(...)
#define LOGOS_FATAL(...)
#endif
