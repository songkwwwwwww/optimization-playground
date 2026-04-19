#include "spsc_queue.h"
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>

using namespace lockfree;

template <typename T>
class SPSCQueueTest : public ::testing::Test {};

using QueueTypes = ::testing::Types<
    SPSCQueueNaive<int, 1024>,
    SPSCQueueAcqRel<int, 1024>,
    SPSCQueuePadded<int, 1024>,
    SPSCQueueCached<int, 1024>
>;

TYPED_TEST_SUITE(SPSCQueueTest, QueueTypes);

TYPED_TEST(SPSCQueueTest, BasicPushPop) {
    TypeParam queue;
    int val;

    EXPECT_FALSE(queue.Pop(val));
    EXPECT_TRUE(queue.Push(1));
    EXPECT_TRUE(queue.Push(2));
    EXPECT_TRUE(queue.Pop(val));
    EXPECT_EQ(val, 1);
    EXPECT_TRUE(queue.Pop(val));
    EXPECT_EQ(val, 2);
    EXPECT_FALSE(queue.Pop(val));
}

TYPED_TEST(SPSCQueueTest, FullEmpty) {
    TypeParam queue; // Capacity 1024

    // Fill it
    for (int i = 0; i < 1023; ++i) {
        EXPECT_TRUE(queue.Push(i));
    }
    EXPECT_FALSE(queue.Push(1023)); // Full

    int val;
    EXPECT_TRUE(queue.Pop(val));
    EXPECT_EQ(val, 0);
    EXPECT_TRUE(queue.Push(1023));
}

TYPED_TEST(SPSCQueueTest, Wraparound) {
    TypeParam queue; // Capacity 1024

    // Fill and drain multiple times to force index wraparound past Capacity.
    for (int round = 0; round < 4; ++round) {
        for (int i = 0; i < 1023; ++i) {
            EXPECT_TRUE(queue.Push(round * 1023 + i));
        }
        EXPECT_FALSE(queue.Push(-1)); // Full

        int val;
        for (int i = 0; i < 1023; ++i) {
            EXPECT_TRUE(queue.Pop(val));
            EXPECT_EQ(val, round * 1023 + i);
        }
        EXPECT_FALSE(queue.Pop(val)); // Empty
    }
}

TYPED_TEST(SPSCQueueTest, MultiThreaded) {
    const int count = 100000;
    TypeParam queue;

    std::thread producer([&]() {
        for (int i = 0; i < count; ++i) {
            while (!queue.Push(i)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (int i = 0; i < count; ++i) {
            int val;
            while (!queue.Pop(val)) {
                std::this_thread::yield();
            }
            EXPECT_EQ(val, i);
        }
    });

    producer.join();
    consumer.join();
}

TYPED_TEST(SPSCQueueTest, MultiThreadedStress) {
    const int count = 1000000;
    TypeParam queue;

    std::thread producer([&]() {
        for (int i = 0; i < count; ++i) {
            while (!queue.Push(i)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (int i = 0; i < count; ++i) {
            int val;
            while (!queue.Pop(val)) {
                std::this_thread::yield();
            }
            EXPECT_EQ(val, i);
        }
    });

    producer.join();
    consumer.join();
}

// --- Move-only type tests (Padded only, to verify move semantics) ---

TEST(SPSCQueueMoveOnly, PushPopUniquePtr) {
    SPSCQueuePadded<std::unique_ptr<int>, 64> queue;

    EXPECT_TRUE(queue.Push(std::make_unique<int>(42)));
    EXPECT_TRUE(queue.Push(std::make_unique<int>(99)));

    std::unique_ptr<int> val;
    EXPECT_TRUE(queue.Pop(val));
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(*val, 42);

    EXPECT_TRUE(queue.Pop(val));
    ASSERT_NE(val, nullptr);
    EXPECT_EQ(*val, 99);

    EXPECT_FALSE(queue.Pop(val));
}

TEST(SPSCQueueMoveOnly, MultiThreadedUniquePtr) {
    const int count = 10000;
    SPSCQueuePadded<std::unique_ptr<int>, 1024> queue;

    std::thread producer([&]() {
        for (int i = 0; i < count; ++i) {
            auto ptr = std::make_unique<int>(i);
            while (!queue.Push(std::move(ptr))) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&]() {
        for (int i = 0; i < count; ++i) {
            std::unique_ptr<int> val;
            while (!queue.Pop(val)) {
                std::this_thread::yield();
            }
            ASSERT_NE(val, nullptr);
            EXPECT_EQ(*val, i);
        }
    });

    producer.join();
    consumer.join();
}
