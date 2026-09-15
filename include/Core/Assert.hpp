#pragma once

#include "Logging.hpp"

#include <cstdlib>
#include <source_location>
#include <string_view>

namespace ml::core::detail {

// This is a hard-failure path, and if a future caller wants to
// turn it into a throw instead of abort() (e.g. for a fuzzing harness), that
// shouldn't require touching the macros.
[[noreturn]] inline void AssertFail(std::string_view expr,
                                    std::source_location loc,
                                    std::string_view msg = "Invalid argument") {
  Logger::GetInstance().Log(LogLevel::Fatal, loc, "Assertion failed: ({}) {}",
                            expr, msg);
  std::abort();
}

template <class... Args>
  requires(sizeof...(Args) > 0)
[[noreturn]] inline void
AssertFail(std::string_view expr, std::source_location loc,
           std::format_string<Args...> format, Args &&...args) {
  AssertFail(expr, loc, std::format(format, std::forward<Args>(args)...));
}

} // namespace ml::core::detail

// Always active, in debug AND release. Use for conditions that must never
// be violated without immediately corrupting program state. If you're ever
// tempted to disable this in release "for performance", that's a sign the
// check belongs in CORE_ASSERT instead, not that VERIFY should be weaker.
#define CORE_VERIFY(cond, ...)                                                 \
  do {                                                                         \
    if (!(cond)) [[unlikely]] {                                                \
      ::ml::core::detail::AssertFail(#cond, std::source_location::current()    \
                                                __VA_OPT__(, ) __VA_ARGS__);   \
    }                                                                          \
  } while (false)

// Compiled out entirely in release - the condition itself is never
// evaluated, so this costs nothing and won't trigger unused-variable
// warnings on release builds. Use for expensive or redundant sanity checks
// you only want while developing (bounds checks, invariant re-verification,
// canary checks on every allocation, etc).
//
// Define CORE_FORCE_ASSERTS to keep these live in a release build too
#if !defined(NDEBUG) || defined(CORE_FORCE_ASSERTS)
#define CORE_ASSERT(cond, ...) CORE_VERIFY(cond __VA_OPT__(, ) __VA_ARGS__)
#define CORE_ENABLE_ASSERTS 1
#else
#define CORE_ASSERT(cond, ...) ((void)0)
#define CORE_ENABLE_ASSERTS 0
#endif

#define UNREACHABLE(...) CORE_ASSERT(false __VA_OPT__(, ) __VA_ARGS__)
