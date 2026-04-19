#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace lockfree {

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_destructive_interference_size;
#else
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

/**
 * Stage 0: Broken (Relaxed-Only)
 *
 * Intentionally incorrect implementation that uses memory_order_relaxed for all
 * operations. On weakly-ordered architectures (ARM64), the consumer may read
 * stale or partially-written buffer data because there is no happens-before
 * relationship between the producer's write and the consumer's read.
 *
 * Use this with ThreadSanitizer (`bazel test --config=tsan ...`) to observe
 * data-race reports.
 *
 * DO NOT use in production.
 */
template <typename T, std::size_t Capacity>
class SPSCQueueBroken {
  static_assert((Capacity & (Capacity - 1)) == 0,
                "Capacity must be a power of two");

 public:
  SPSCQueueBroken() : head_(0), tail_(0) {}

  template <typename U>
  bool Push(U&& value) noexcept(std::is_nothrow_assignable_v<T&, U&&>) {
    const size_t h = head_.load(std::memory_order_relaxed);
    const size_t t = tail_.load(std::memory_order_relaxed);
    if (Next(h) == t) return false;
    buffer_[h] = std::forward<U>(value);
    head_.store(Next(h), std::memory_order_relaxed);
    return true;
  }

  bool Pop(T& value) noexcept(std::is_nothrow_assignable_v<T&, T&&>) {
    const size_t h = head_.load(std::memory_order_relaxed);
    const size_t t = tail_.load(std::memory_order_relaxed);
    if (h == t) return false;
    value = std::move(buffer_[t]);
    tail_.store(Next(t), std::memory_order_relaxed);
    return true;
  }

 private:
  size_t Next(size_t i) const noexcept { return (i + 1) & (Capacity - 1); }

  T buffer_[Capacity];
  std::atomic<size_t> head_;
  std::atomic<size_t> tail_;
};

/**
 * Stage 1: Naive (SeqCst, No Padding)
 *
 * Uses default memory_order_seq_cst for all atomic operations. head_ and tail_
 * are placed right after buffer_ with no alignment directive, so they will
 * likely reside on the same cache line — exhibiting false sharing between the
 * producer (writes head_) and consumer (writes tail_).
 */
template <typename T, std::size_t Capacity>
class SPSCQueueNaive {
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

 public:
  SPSCQueueNaive() : head_(0), tail_(0) {}

  template <typename U>
  bool Push(U&& value) noexcept(std::is_nothrow_assignable_v<T&, U&&>) {
    const size_t h = head_.load();
    const size_t t = tail_.load();
    if (Next(h) == t) return false;
    buffer_[h] = std::forward<U>(value);
    head_.store(Next(h));
    return true;
  }

  bool Pop(T& value) noexcept(std::is_nothrow_assignable_v<T&, T&&>) {
    const size_t h = head_.load();
    const size_t t = tail_.load();
    if (h == t) return false;
    value = std::move(buffer_[t]);
    tail_.store(Next(t));
    return true;
  }

 private:
  size_t Next(size_t i) const noexcept { return (i + 1) & (Capacity - 1); }

  // head_ and tail_ intentionally unpadded: they likely share a cache line,
  // causing false sharing between producer and consumer cores.
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

  template <typename U>
  bool Push(U&& value) noexcept(std::is_nothrow_assignable_v<T&, U&&>) {
    const size_t h = head_.load(std::memory_order_relaxed);
    const size_t t = tail_.load(std::memory_order_acquire);
    if (Next(h) == t) return false;
    buffer_[h] = std::forward<U>(value);
    head_.store(Next(h), std::memory_order_release);
    return true;
  }

  bool Pop(T& value) noexcept(std::is_nothrow_assignable_v<T&, T&&>) {
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
 *
 * Eliminates false sharing by placing head_, tail_, and buffer_ on separate
 * cache lines. Note: member order differs from Naive/AcqRel (indices before
 * buffer) so that alignas applies cleanly to each atomic without the buffer
 * straddling them.
 */
template <typename T, std::size_t Capacity>
class SPSCQueuePadded {
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

 public:
  SPSCQueuePadded() : head_(0), tail_(0) {}

  template <typename U>
  bool Push(U&& value) noexcept(std::is_nothrow_assignable_v<T&, U&&>) {
    const size_t h = head_.load(std::memory_order_relaxed);
    const size_t t = tail_.load(std::memory_order_acquire);
    if (Next(h) == t) return false;
    buffer_[h] = std::forward<U>(value);
    head_.store(Next(h), std::memory_order_release);
    return true;
  }

  bool Pop(T& value) noexcept(std::is_nothrow_assignable_v<T&, T&&>) {
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
 *
 * NOTE: This optimization can cause significant performance regression (up to 5-6x slower)
 * on Unified Memory Architectures like Apple Silicon (ARM64) where shared L2/L3 caches
 * make atomic loads very efficient, while extra branching and logic introduce stalls.
 */
template <typename T, std::size_t Capacity>
class SPSCQueueCached {
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

 public:
  SPSCQueueCached() {
    producer_.head.store(0);
    producer_.tail_cache = 0;
    consumer_.tail.store(0);
    consumer_.head_cache = 0;
  }

  template <typename U>
  bool Push(U&& value) noexcept(std::is_nothrow_assignable_v<T&, U&&>) {
    const size_t h = producer_.head.load(std::memory_order_relaxed);
    if (Next(h) == producer_.tail_cache) {
      producer_.tail_cache = consumer_.tail.load(std::memory_order_acquire);
      if (Next(h) == producer_.tail_cache) return false;
    }
    buffer_[h] = std::forward<U>(value);
    producer_.head.store(Next(h), std::memory_order_release);
    return true;
  }

  bool Pop(T& value) noexcept(std::is_nothrow_assignable_v<T&, T&&>) {
    const size_t t = consumer_.tail.load(std::memory_order_relaxed);
    if (consumer_.head_cache == t) {
      consumer_.head_cache = producer_.head.load(std::memory_order_acquire);
      if (consumer_.head_cache == t) return false;
    }
    value = std::move(buffer_[t]);
    consumer_.tail.store(Next(t), std::memory_order_release);
    return true;
  }

 private:
  size_t Next(size_t i) const noexcept { return (i + 1) & (Capacity - 1); }

  // Producer-owned cache line
  struct alignas(hardware_destructive_interference_size) ProducerData {
    std::atomic<size_t> head;
    size_t tail_cache;
  } producer_;

  // Consumer-owned cache line
  struct alignas(hardware_destructive_interference_size) ConsumerData {
    std::atomic<size_t> tail;
    size_t head_cache;
  } consumer_;

  // Buffer on separate cache line
  alignas(hardware_destructive_interference_size) T buffer_[Capacity];
};

/**
 * Recommended implementation for general use.
 */
template <typename T, std::size_t Capacity>
using SPSCQueue = SPSCQueuePadded<T, Capacity>;

} // namespace lockfree
