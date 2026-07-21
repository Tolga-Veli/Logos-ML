#pragma once

#include <memory>

namespace ml::core {
template <class T> class Storage {
public:
  explicit Storage(size_t size)
      : m_Data(std::make_unique<T[]>(size)), m_Size(size) {}

  [[nodiscard]] T *data() { return m_Data.get(); }
  [[nodiscard]] const T *data() const { return m_Data.get(); }
  [[nodiscard]] size_t size() const { return m_Size; }

private:
  std::unique_ptr<T[]> m_Data;
  size_t m_Size;
};
} // namespace ml::core
