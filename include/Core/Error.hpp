#pragma once

#include <concepts>
#include <format>
#include <source_location>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ml {

namespace detail {
template <class... Args> struct ErrorFormat {
  std::format_string<Args...> fmt;
  std::source_location location;

  template <class S>
    requires(!std::same_as<std::remove_cvref_t<S>, ErrorFormat>)
  consteval ErrorFormat(S &&fmt, std::source_location location = std::source_location::current())
      : fmt(std::forward<S>(fmt)), location(location) {}
};

template <class... Args> [[nodiscard]] std::string FormatError(const ErrorFormat<Args...> &format, Args &&...args) {

  return std::format("{}:{}: {}", format.location.file_name(), format.location.line(),
                     std::format(format.fmt, std::forward<Args>(args)...));
}
} // namespace detail

class Error : public std::runtime_error {
public:
  [[nodiscard]] const std::source_location &location() const noexcept { return m_Location; }

protected:
  template <class... Args> Error(const detail::ErrorFormat<Args...> format, Args &&...args)
      : std::runtime_error(detail::FormatError(format, std::forward<Args>(args)...)), m_Location(format.location) {}

private:
  std::source_location m_Location;
};

class ShapeError final : public Error {
public:
  template <class... Args> ShapeError(detail::ErrorFormat<std::type_identity_t<Args>...> format, Args &&...args)
      : Error(format, std::forward<Args>(args)...) {}
};

class DTypeError final : public Error {
public:
  template <class... Args> DTypeError(detail::ErrorFormat<std::type_identity_t<Args>...> format, Args &&...args)
      : Error(format, std::forward<Args>(args)...) {}
};

class DeviceError final : public Error {
public:
  template <class... Args> DeviceError(detail::ErrorFormat<std::type_identity_t<Args>...> format, Args &&...args)
      : Error(format, std::forward<Args>(args)...) {}
};

} // namespace ml
