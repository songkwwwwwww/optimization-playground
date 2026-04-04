#include "spsc_queue.h"
#include <benchmark/benchmark.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <memory>

using namespace lockfree;

// =============================================================================
// Throughput benchmark
//
// Single-threaded from Google Benchmark's perspective. Each iteration spawns
// a consumer thread and joins it, guaranteeing that producer and consumer
// always process the same number of items (no deadlock).
//
// The queue is heap-allocated once and reused across iterations to avoid
// measuring allocation / alignas padding overhead.
// =============================================================================

template <typename Q>
void BM_SPSC_Throughput(benchmark::State& state) {
    const int count = state.range(0);
    auto queue = std::make_unique<Q>();

    for (auto _ : state) {
        std::thread consumer([&]() {
            for (int i = 0; i < count; ++i) {
                int val;
                while (!queue->Pop(val));
            }
        });

        for (int i = 0; i < count; ++i) {
            while (!queue->Push(i));
        }

        consumer.join();
    }
    state.SetItemsProcessed(state.iterations() * count);
}

#define BENCHMARK_THROUGHPUT(QueueType, Cap, Items)                 \
    BENCHMARK_TEMPLATE(BM_SPSC_Throughput, QueueType<int, Cap>)    \
        ->Arg(Items)                                               \
        ->Unit(benchmark::kMillisecond)                            \
        ->Name(#QueueType "/" #Cap "/throughput");

// Primary comparison across optimization stages (65536 capacity).
BENCHMARK_THROUGHPUT(SPSCQueueNaive,  65536, 1000000);
BENCHMARK_THROUGHPUT(SPSCQueueAcqRel, 65536, 1000000);
BENCHMARK_THROUGHPUT(SPSCQueuePadded, 65536, 1000000);
BENCHMARK_THROUGHPUT(SPSCQueueCached, 65536, 1000000);

// Capacity variation — shows cache behavior differences.
BENCHMARK_THROUGHPUT(SPSCQueuePadded, 64,      100000);
BENCHMARK_THROUGHPUT(SPSCQueuePadded, 256,     100000);
BENCHMARK_THROUGHPUT(SPSCQueuePadded, 1048576, 1000000);

// =============================================================================
// Latency benchmark (round-trip time per item)
// =============================================================================

struct TimedItem {
    std::chrono::steady_clock::time_point push_time;
};

template <typename Q>
void BM_SPSC_Latency(benchmark::State& state) {
    static constexpr int kItems = 10000;
    using Clock = std::chrono::steady_clock;

    auto queue = std::make_unique<Q>();
    std::vector<int64_t> latencies;
    latencies.reserve(kItems);

    for (auto _ : state) {
        latencies.clear();

        std::thread consumer([&]() {
            for (int i = 0; i < kItems; ++i) {
                TimedItem item;
                while (!queue->Pop(item));
                auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - item.push_time).count();
                latencies.push_back(ns);
            }
        });

        for (int i = 0; i < kItems; ++i) {
            TimedItem item{Clock::now()};
            while (!queue->Push(std::move(item)));
        }

        consumer.join();
    }

    // Report percentiles from the last iteration.
    std::sort(latencies.begin(), latencies.end());
    if (!latencies.empty()) {
        auto percentile = [&](double p) -> int64_t {
            size_t idx = static_cast<size_t>(
                p * static_cast<double>(latencies.size() - 1));
            return latencies[idx];
        };
        state.counters["p50_ns"] = benchmark::Counter(
            static_cast<double>(percentile(0.50)),
            benchmark::Counter::kDefaults);
        state.counters["p99_ns"] = benchmark::Counter(
            static_cast<double>(percentile(0.99)),
            benchmark::Counter::kDefaults);
        state.counters["p999_ns"] = benchmark::Counter(
            static_cast<double>(percentile(0.999)),
            benchmark::Counter::kDefaults);
    }
}

#define BENCHMARK_LATENCY(QueueType, Cap)                                     \
    BENCHMARK_TEMPLATE(BM_SPSC_Latency, QueueType<TimedItem, Cap>)            \
        ->Unit(benchmark::kMillisecond)                                       \
        ->Name(#QueueType "/" #Cap "/latency");

BENCHMARK_LATENCY(SPSCQueuePadded, 1024);
BENCHMARK_LATENCY(SPSCQueueCached, 1024);

BENCHMARK_MAIN();
