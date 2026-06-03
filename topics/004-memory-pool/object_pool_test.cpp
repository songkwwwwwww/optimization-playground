#include "object_pool.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace memory_pool {
namespace {

struct Counted {
  explicit Counted(int value_arg) : value(value_arg) { ++constructed; }
  Counted(const Counted&) = delete;
  Counted& operator=(const Counted&) = delete;
  ~Counted() { ++destroyed; }

  int value;
  static inline int constructed = 0;
  static inline int destroyed = 0;
};

struct alignas(64) CacheLineAligned {
  explicit CacheLineAligned(std::uint64_t value_arg) : value(value_arg) {}

  std::uint64_t value;
};

TEST(ObjectPoolTest, ConstructsAndDestroysObjects) {
  Counted::constructed = 0;
  Counted::destroyed = 0;

  ObjectPool<Counted, 4> pool;
  Counted* first = pool.New(10);
  Counted* second = pool.New(20);

  EXPECT_EQ(first->value, 10);
  EXPECT_EQ(second->value, 20);
  EXPECT_EQ(pool.live_count(), 2);
  EXPECT_EQ(pool.free_count(), 2);
  EXPECT_EQ(pool.block_count(), 1);
  EXPECT_EQ(Counted::constructed, 2);
  EXPECT_EQ(Counted::destroyed, 0);

  pool.Delete(first);
  pool.Delete(second);

  EXPECT_EQ(pool.live_count(), 0);
  EXPECT_EQ(pool.free_count(), 4);
  EXPECT_EQ(Counted::destroyed, 2);
}

TEST(ObjectPoolTest, ReusesFreedSlots) {
  ObjectPool<int, 2> pool;
  int* first = pool.New(1);
  int* second = pool.New(2);

  pool.Delete(second);
  int* reused = pool.New(3);

  EXPECT_EQ(reused, second);
  EXPECT_EQ(*reused, 3);
  EXPECT_EQ(pool.live_count(), 2);
  EXPECT_EQ(pool.block_count(), 1);

  pool.Delete(first);
  pool.Delete(reused);
}

TEST(ObjectPoolTest, GrowsByBlocks) {
  ObjectPool<int, 3> pool;
  std::vector<int*> values;
  values.reserve(7);

  for (int i = 0; i < 7; ++i) {
    values.push_back(pool.New(i));
  }

  EXPECT_EQ(pool.live_count(), 7);
  EXPECT_EQ(pool.block_count(), 3);
  EXPECT_EQ(pool.capacity(), 9);
  EXPECT_EQ(pool.free_count(), 2);

  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(*values[i], i);
    pool.Delete(values[i]);
  }
}

TEST(ObjectPoolTest, ClearDestroysLiveObjects) {
  Counted::constructed = 0;
  Counted::destroyed = 0;

  ObjectPool<Counted, 2> pool;
  pool.New(1);
  Counted* deleted = pool.New(2);
  pool.New(3);

  pool.Delete(deleted);
  EXPECT_EQ(Counted::destroyed, 1);

  pool.Clear();

  EXPECT_EQ(Counted::constructed, 3);
  EXPECT_EQ(Counted::destroyed, 3);
  EXPECT_EQ(pool.live_count(), 0);
  EXPECT_EQ(pool.free_count(), 0);
  EXPECT_EQ(pool.block_count(), 0);
}

TEST(ObjectPoolTest, SupportsMoveOnlyObjects) {
  ObjectPool<std::unique_ptr<int>, 8> pool;
  std::unique_ptr<int>* value = pool.New(std::make_unique<int>(42));

  ASSERT_NE(*value, nullptr);
  EXPECT_EQ(**value, 42);

  pool.Delete(value);
  EXPECT_EQ(pool.live_count(), 0);
}

TEST(ObjectPoolTest, PreservesOverAlignedObjectAlignment) {
  ObjectPool<CacheLineAligned, 4> pool;
  CacheLineAligned* value = pool.New(123);
  auto address = reinterpret_cast<std::uintptr_t>(value);

  EXPECT_EQ(address % alignof(CacheLineAligned), 0);
  EXPECT_EQ(value->value, 123);

  pool.Delete(value);
}

}  // namespace
}  // namespace memory_pool
