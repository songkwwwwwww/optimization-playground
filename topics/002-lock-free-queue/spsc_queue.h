#pragma once

#include <atomic>
#include <cstddef>
#include <new>

namespace lockfree {

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_destructive_interference_size;
#else
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

/**
 * Stage 1: Naive (SeqCst, No Padding)
 */
template <typename T, std::size_t Capacity>
class SPSCQueueNaive {
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

 public:
  SPSCQueueNaive() : head_(0), tail_(0) {}

  bool Push(const T& value) noexcept {
    const size_t h = head_.load();
    const size_t t = tail_.load();
    if (Next(h) == t) return false;
    buffer_[h] = value;
    head_.store(Next(h));
    return true;
  }

  bool Push(T&& value) noexcept {
    const size_t h = head_.load();
    const size_t t = tail_.load();
    if (Next(h) == t) return false;
    buffer_[h] = std::move(value);
    head_.store(Next(h));
    return true;
  }

  bool Pop(T& value) noexcept {
    const size_t h = head_.load();
    const size_t t = tail_.load();
    if (h == t) return false;
    value = std::move(buffer_[t]);
    tail_.store(Next(t));
    return true;
  }

 private:
  size_t Next(size_t i) const noexcept { return (i + 1) & (Capacity - 1); }

  T buffer_[Capacity];
  std::atomic<size_t> head_;
  std::atomic<size_t> tail_;
};

/**
 * Stage 2: Acquire-Release (No Padding)
 */
template <typename T, std::size_t Capacity>
class SPSCQueueAcqRel {
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

 public:
  SPSCQueueAcqRel() : head_(0), tail_(0) {}

  bool Push(const T& value) noexcept {
    const size_t h = head_.load(std::memory_order_relaxed);
    const size_t t = tail_.load(std::memory_order_acquire);
    if (Next(h) == t) return false;
    buffer_[h] = value;
    head_.store(Next(h), std::memory_order_release);
    return true;
  }

  bool Push(T&& value) noexcept {
    const size_t h = head_.load(std::memory_order_relaxed);
    const size_t t = tail_.load(std::memory_order_acquire);
    if (Next(h) == t) return false;
    buffer_[h] = std::move(value);
    head_.store(Next(h), std::memory_order_release);
    return true;
  }

  bool Pop(T& value) noexcept {
    const size_t h = head_.load(std::memory_order_acquire);
    const size_t t = tail_.load(std::memory_order_relaxed);
    if (h == t) return false;
    value = std::move(buffer_[t]);
    tail_.store(Next(t), std::memory_order_release);
    return true;
  }

 private:
  size_t Next(size_t i) const noexcept { return (i + 1) & (Capacity - 1); }

  T buffer_[Capacity];
  std::atomic<size_t> head_;
  std::atomic<size_t> tail_;
};

/**
 * Stage 3: Padded (Acquire-Release + Alignment)
 */
template <typename T, std::size_t Capacity>
class SPSCQueuePadded {
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

 public:
  SPSCQueuePadded() : head_(0), tail_(0) {}

  bool Push(const T& value) noexcept {
    const size_t h = head_.load(std::memory_order_relaxed);
    const size_t t = tail_.load(std::memory_order_acquire);
    if (Next(h) == t) return false;
    buffer_[h] = value;
    head_.store(Next(h), std::memory_order_release);
    return true;
  }

  bool Push(T&& value) noexcept {
    const size_t h = head_.load(std::memory_order_relaxed);
    const size_t t = tail_.load(std::memory_order_acquire);
    if (Next(h) == t) return false;
    buffer_[h] = std::move(value);
    head_.store(Next(h), std::memory_order_release);
    return true;
  }

  bool Pop(T& value) noexcept {
    const size_t h = head_.load(std::memory_order_acquire);
    const size_t t = tail_.load(std::memory_order_relaxed);
    if (h == t) return false;
    value = std::move(buffer_[t]);
    tail_.store(Next(t), std::memory_order_release);
    return true;
  }

 private:
  size_t Next(size_t i) const noexcept { return (i + 1) & (Capacity - 1); }

  alignas(hardware_destructive_interference_size) std::atomic<size_t> head_;
  alignas(hardware_destructive_interference_size) std::atomic<size_t> tail_;
  alignas(hardware_destructive_interference_size) T buffer_[Capacity];
};

/**
 * Stage 4: Cached (The "Outrageous" Optimization)
 */
template <typename T, std::size_t Capacity>
class SPSCQueueCached {
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

 public:
  SPSCQueueCached() : head_(0), tail_cache_(0), tail_(0), head_cache_(0) {}

  bool Push(const T& value) noexcept {
    const size_t h = head_.load(std::memory_order_relaxed);
    if (Next(h) == tail_cache_) {
      tail_cache_ = tail_.load(std::memory_order_acquire);
      if (Next(h) == tail_cache_) return false;
    }
    buffer_[h] = value;
    head_.store(Next(h), std::memory_order_release);
    return true;
  }

  bool Push(T&& value) noexcept {
    const size_t h = head_.load(std::memory_order_relaxed);
    if (Next(h) == tail_cache_) {
      tail_cache_ = tail_.load(std::memory_order_acquire);
      if (Next(h) == tail_cache_) return false;
    }
    buffer_[h] = std::move(value);
    head_.store(Next(h), std::memory_order_release);
    return true;
  }

  bool Pop(T& value) noexcept {
    const size_t t = tail_.load(std::memory_order_relaxed);
    if (head_cache_ == t) {
      head_cache_ = head_.load(std::memory_order_acquire);
      if (head_cache_ == t) return false;
    }
    value = std::move(buffer_[t]);
    tail_.store(Next(t), std::memory_order_release);
    return true;
  }

 private:
  size_t Next(size_t i) const noexcept { return (i + 1) & (Capacity - 1); }

  // Producer-owned cache line
  alignas(hardware_destructive_interference_size) std::atomic<size_t> head_;
  size_t tail_cache_;

  // Consumer-owned cache line
  alignas(hardware_destructive_interference_size) std::atomic<size_t> tail_;
  size_t head_cache_;

  // Buffer on separate cache line
  alignas(hardware_destructive_interference_size) T buffer_[Capacity];
};

/**
 * Recommended implementation for general use.
 */
template <typename T, std::size_t Capacity>
using SPSCQueue = SPSCQueuePadded<T, Capacity>;

} // namespace lockfree
