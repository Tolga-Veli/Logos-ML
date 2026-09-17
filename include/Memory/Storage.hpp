#pragma once

#include <cstring>
#include <memory>

#include "Memory/Device.hpp"
#include "Memory/IntrusiveRef.hpp"

namespace ml::memory {
class Storage final : public detail::IntrusiveRefCounted {
public:
  Storage() noexcept = default;
  ~Storage() noexcept = default;

  explicit Storage(std::size_t bytes, Device device);

  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;

  Storage(Storage &&other) noexcept;
  Storage &operator=(Storage &&other) noexcept;

  [[nodiscard]] std::byte *raw_data() noexcept { return m_Data.get(); }
  [[nodiscard]] const std::byte *raw_data() const noexcept { return m_Data.get(); }

  template <class T> [[nodiscard]] T *data() noexcept { return reinterpret_cast<T *>(m_Data.get()); }
  template <class T> const T *data() const noexcept { return reinterpret_cast<const T *>(m_Data.get()); }

  [[nodiscard]] std::size_t size_bytes() const noexcept { return m_Bytes; }
  [[nodiscard]] Device device() const noexcept { return m_Device; }
  [[nodiscard]] bool empty() const noexcept { return m_Data == nullptr; }
  [[nodiscard]] explicit operator bool() const noexcept { return m_Data != nullptr; }

  void reserve(std::size_t bytes);

private:
  std::size_t m_Bytes{};
  Device m_Device{};
  std::unique_ptr<std::byte, detail::DeviceDeleter> m_Data{nullptr, detail::DeviceDeleter{Device{}}};
};
} // namespace ml::memory
