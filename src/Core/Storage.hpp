#pragma once

#include <cassert>
#include <memory>

namespace ml::core {
template <class T> class Storage {
public:
  explicit Storage(int size)
      : m_Data(std::make_unique<T[]>(size)), m_Size(size) {}

  [[nodiscard]] T *data() { return m_Data.get(); }
  [[nodiscard]] const T *data() const { return m_Data.get(); }

  [[nodiscard]] bool empty() const noexcept { return m_Size == 0; }

  [[nodiscard]] int size() const { return m_Size; }

  [[nodiscard]] T &operator[](int index) {
    assert(index < m_Size);
    return m_Data[index];
  }

private:
  std::unique_ptr<T[]> m_Data;
  int m_Size;
};
} // namespace ml::core
