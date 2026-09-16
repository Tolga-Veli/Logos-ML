#pragma once

#include <concepts>
#include <cstdint>
#include <utility>

#include "Core/Assert.hpp"

namespace ml::memory {

namespace detail {
// IntrusiveRef<T> must retain the concrete type through which the object must be deleted since the base destructor isn't virtual
// NOTE: NOT THREAD SAFE
class IntrusiveRefCounted {
public:
  IntrusiveRefCounted() noexcept = default;

  IntrusiveRefCounted(const IntrusiveRefCounted &) = delete;
  IntrusiveRefCounted &operator=(const IntrusiveRefCounted &) = delete;
  IntrusiveRefCounted(IntrusiveRefCounted &&) = delete;
  IntrusiveRefCounted &operator=(IntrusiveRefCounted &&) = delete;

  void add_ref() noexcept { ++m_Refs; }

  [[nodiscard]] bool release_ref() noexcept {
    CORE_ASSERT(m_Refs > 0, "Refs was 0");
    return --m_Refs == 0;
  }

  [[nodiscard]] std::uint32_t refs() const noexcept { return m_Refs; }

protected:
  ~IntrusiveRefCounted() = default;

private:
  std::uint32_t m_Refs{};
};
} // namespace detail

template <class T>
concept IntrusiveRefBase = std::derived_from<T, detail::IntrusiveRefCounted>;

template <IntrusiveRefBase T> class IntrusiveRef {
public:
  IntrusiveRef() noexcept = default;
  ~IntrusiveRef() noexcept { release(); }

  explicit IntrusiveRef(T *ptr) noexcept;

  IntrusiveRef(const IntrusiveRef &other) noexcept;
  IntrusiveRef &operator=(const IntrusiveRef &other) noexcept;
  IntrusiveRef(IntrusiveRef &&other) noexcept;
  IntrusiveRef &operator=(IntrusiveRef &&other) noexcept;

  [[nodiscard]] T *get() const noexcept { return m_Ptr; }
  [[nodiscard]] T *operator->() const noexcept { return m_Ptr; }
  [[nodiscard]] T &operator*() const noexcept { return *m_Ptr; }
  [[nodiscard]] explicit operator bool() const noexcept { return m_Ptr != nullptr; }

  void reset() noexcept;

private:
  T *m_Ptr{};

  void release() noexcept;
};

template <IntrusiveRefBase T, class... Args> [[nodiscard]] IntrusiveRef<T> CreateIntrusiveRef(Args &&...args) {
  return IntrusiveRef<T>(new T(std::forward<Args>(args)...));
}
} // namespace ml::memory

#include "Memory/IntrusiveRef.inl"
