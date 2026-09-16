#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <new>

#include "Core/Assert.hpp"

namespace ml::memory {
enum class DeviceType : std::uint8_t { CPU };

class Device {
public:
  constexpr Device(DeviceType type = DeviceType::CPU) noexcept : m_Type(type) {}

  [[nodiscard]] constexpr DeviceType type() const noexcept { return m_Type; }

  bool operator<=>(const Device &) const noexcept = default;

private:
  DeviceType m_Type;
};

[[nodiscard]] constexpr std::string_view to_string(Device device) {
  switch (device.type()) {
  case DeviceType::CPU:
    return "CPU";
  }

  UNREACHABLE("Unknown device");
  return "Unknown";
}

namespace detail {
inline constexpr std::size_t DEFAULT_CPU_ALIGNMENT{64};

[[nodiscard]] inline std::byte *Allocate(std::size_t bytes, Device device,
                                         std::size_t alignment = DEFAULT_CPU_ALIGNMENT) {
  switch (device.type()) {
  case DeviceType::CPU:
    return static_cast<std::byte *>(::operator new(bytes, std::align_val_t{alignment}));
  }

  return nullptr;
}

struct DeviceDeleter {
  Device device;
  std::size_t alignment;

  DeviceDeleter(Device _device, std::size_t _alignment = DEFAULT_CPU_ALIGNMENT)
      : device(_device), alignment(_alignment) {}

  void operator()(std::byte *ptr) const noexcept {
    if (!ptr)
      return;

    switch (device.type()) {
    case DeviceType::CPU:
      ::operator delete(ptr, std::align_val_t{alignment});
      break;
    }
  }
};
} // namespace detail
} // namespace ml::memory

namespace std {
template <> struct formatter<ml::memory::Device, char> : formatter<string_view, char> {
  template <class FormatContext> auto format(ml::memory::Device device, FormatContext &ctx) const {
    return formatter<string_view, char>::format(ml::memory::to_string(device), ctx);
  }
};
} // namespace std
