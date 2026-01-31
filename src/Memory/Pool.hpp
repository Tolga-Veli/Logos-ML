#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

#include "AlignedAlloc.hpp"

namespace ob::Memory {
template <class T> class Pool {
public:
  Pool() = delete;
  Pool(std::size_t capacity)
      : m_Capacity(capacity), m_Free(nullptr), m_Nodes(nullptr) {
    if (capacity == 0)
      throw std::bad_alloc();

    m_Nodes = static_cast<Node *>(
        aligned_malloc(sizeof(Node) * capacity, alignof(Node)));
    if (!m_Nodes)
      throw std::bad_alloc();

    for (std::size_t i = 0; i < capacity - 1; i++)
      m_Nodes[i].next = &m_Nodes[i + 1];

    m_Nodes[capacity - 1].next = nullptr;
    m_Free = &m_Nodes[0];
  }

  ~Pool() { aligned_free(m_Nodes); }

  Pool(const Pool &) = delete;
  Pool &operator=(const Pool &) = delete;

  Pool(Pool &&other) noexcept
      : m_Capacity(other.m_Capacity), m_Free(other.m_Free),
        m_Nodes(other.m_Nodes) {
    other.m_Capacity = 0;
    other.m_Free = nullptr;
    other.m_Nodes = nullptr;
  }

  Pool &operator=(Pool &&other) noexcept {
    if (this == &other)
      return *this;

    aligned_free(m_Nodes);

    m_Capacity = other.m_Capacity;
    m_Free = other.m_Free;
    m_Nodes = other.m_Nodes;

    other.m_Capacity = 0;
    other.m_Free = nullptr;
    other.m_Nodes = nullptr;
    return *this;
  }

  T *allocate() {
    if (!m_Free)
      throw std::bad_alloc();
    Node *node = m_Free;
    m_Free = node->next;
    return reinterpret_cast<T *>(node->storage);
  }

  void deallocate(T *ptr) noexcept {
    if (!ptr)
      return;

    assert(owns(ptr) == true &&
           "The pointer to be deallocated is not owned by the pool");

    constexpr std::size_t offset = offsetof(Node, storage);
    Node *node =
        reinterpret_cast<Node *>(reinterpret_cast<std::byte *>(ptr) - offset);
    node->next = m_Free;
    m_Free = node;
  }

  template <class... Args> T *allocate_and_create(Args &&...args) {
    T *ptr = allocate();
    // if constructor throws and the program halts I leak a node
    // that's why I have to have a try, catch block to catch the exceoption and
    // release the memory I had previously
    try {
      ::new ((void *)ptr) T(std::forward<Args>(args)...);
    } catch (...) {
      deallocate(ptr);
      throw;
    }
    return ptr;
  }

  void destroy(T *ptr) noexcept {
    if (!ptr)
      return;
    ptr->~T();
    deallocate(ptr);
  }

private:
  struct Node {
    Node *next;
    alignas(T) std::byte storage[sizeof(T)];
  };
  std::size_t m_Capacity;
  Node *m_Free, *m_Nodes;

  bool owns(const T *ptr) const noexcept {
    if (!ptr || !m_Nodes || m_Capacity == 0)
      return false;

    const auto base = reinterpret_cast<const std::byte *>(m_Nodes);
    const auto end = base + m_Capacity * sizeof(Node);

    const auto pb = reinterpret_cast<const std::byte *>(ptr);
    constexpr std::size_t off = offsetof(Node, storage);

    if (pb < base + off || pb >= end)
      return false;

    const auto nb = pb - off;
    const auto diff = static_cast<std::size_t>(nb - base);

    if (diff % sizeof(Node) != 0)
      return false;
    if (reinterpret_cast<std::uintptr_t>(pb) % alignof(T) != 0)
      return false;

    return true;
  }
};

template <class T> class PoolAllocator {
  // the allocator is expected to allocate at most one thing at a time
public:
  using value_type = T;

  PoolAllocator() noexcept : m_Pool(nullptr) {}
  explicit PoolAllocator(Pool<T> &pool) noexcept : m_Pool(&pool) {}

  template <class U> PoolAllocator(const PoolAllocator<U> &) = delete;

  T *allocate(std::size_t count) {
    if (!m_Pool)
      throw std::bad_alloc();
    return m_Pool->allocate();
  }

  void deallocate(T *ptr) noexcept {
    if (!ptr)
      return;
    if (m_Pool)
      m_Pool->deallocate(ptr);
  }

  template <class U> struct rebind {
    using other = PoolAllocator<U>;
  };

private:
  Pool<T> *m_Pool;
};
} // namespace ob::Memory
