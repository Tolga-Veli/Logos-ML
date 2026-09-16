#include "Memory/Storage.hpp"

namespace ml::memory {
Storage::Storage(std::size_t bytes, Device device)
    : m_Bytes(bytes), m_Device(device), m_Data(detail::Allocate(m_Bytes, m_Device), detail::DeviceDeleter(m_Device)) {}

Storage::Storage(Storage &&other) noexcept
    : m_Bytes(std::exchange(other.m_Bytes, 0)), m_Device(other.m_Device), m_Data(std::move(other.m_Data)) {}

Storage &Storage::operator=(Storage &&other) noexcept {
  if (this == &other)
    return *this;

  m_Bytes = std::exchange(other.m_Bytes, 0);
  m_Device = other.m_Device;
  m_Data = std::move(other.m_Data);

  return *this;
}

void Storage::reserve(std::size_t bytes) {
  if (m_Bytes >= bytes)
    return;

  auto data = std::unique_ptr<std::byte, detail::DeviceDeleter>(detail::Allocate(bytes, m_Device),
                                                                detail::DeviceDeleter(m_Device));

  std::memcpy(data.get(), m_Data.get(), m_Bytes);
  m_Data = std::move(data);
  m_Bytes = bytes;
}

} // namespace ml::memory
