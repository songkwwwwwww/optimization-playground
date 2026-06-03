#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace memory_pool {

template <typename T, std::size_t BlockCapacity = 1024>
class ObjectPool {
  static_assert(BlockCapacity > 0, "BlockCapacity must be greater than zero");
  static_assert(!std::is_const_v<T>, "ObjectPool<T> cannot store const T");
  static_assert(!std::is_reference_v<T>,
                "ObjectPool<T> cannot store references");

 public:
  ObjectPool() = default;

  ObjectPool(const ObjectPool&) = delete;
  ObjectPool& operator=(const ObjectPool&) = delete;

  ObjectPool(ObjectPool&&) = delete;
  ObjectPool& operator=(ObjectPool&&) = delete;

  ~ObjectPool() { Clear(); }

  template <typename... Args>
  T* New(Args&&... args) {
    if (free_head_ == nullptr) {
      AddBlock();
    }

    Slot* slot = free_head_;
    free_head_ = free_head_->next;

    T* object = std::construct_at(reinterpret_cast<T*>(slot->storage),
                                  std::forward<Args>(args)...);
    slot->occupied = true;
    ++live_count_;
    --free_count_;
    return object;
  }

  void Delete(T* object) noexcept(std::is_nothrow_destructible_v<T>) {
    if (object == nullptr) {
      return;
    }

    Slot* slot = reinterpret_cast<Slot*>(object);
    std::destroy_at(object);
    slot->occupied = false;
    slot->next = free_head_;
    free_head_ = slot;
    --live_count_;
    ++free_count_;
  }

  void Clear() noexcept(std::is_nothrow_destructible_v<T>) {
    for (const std::unique_ptr<Block>& block : blocks_) {
      for (Slot& slot : block->slots) {
        if (slot.occupied) {
          std::destroy_at(reinterpret_cast<T*>(slot.storage));
          slot.occupied = false;
        }
      }
    }

    blocks_.clear();
    free_head_ = nullptr;
    live_count_ = 0;
    free_count_ = 0;
  }

  std::size_t live_count() const noexcept { return live_count_; }
  std::size_t free_count() const noexcept { return free_count_; }
  std::size_t block_count() const noexcept { return blocks_.size(); }
  std::size_t capacity() const noexcept {
    return block_count() * BlockCapacity;
  }

 private:
  static constexpr std::size_t kSlotAlignment = alignof(T) > alignof(void*)
                                                    ? alignof(T)
                                                    : alignof(void*);

  struct alignas(kSlotAlignment) Slot {
    Slot() : next(nullptr), occupied(false) {}

    union {
      Slot* next;
      std::byte storage[sizeof(T)];
    };
    bool occupied;
  };

  struct Block {
    std::array<Slot, BlockCapacity> slots;
  };

  void AddBlock() {
    auto block = std::make_unique<Block>();
    for (Slot& slot : block->slots) {
      slot.next = free_head_;
      free_head_ = &slot;
    }
    blocks_.push_back(std::move(block));
    free_count_ += BlockCapacity;
  }

  std::vector<std::unique_ptr<Block>> blocks_;
  Slot* free_head_ = nullptr;
  std::size_t live_count_ = 0;
  std::size_t free_count_ = 0;
};

}  // namespace memory_pool
