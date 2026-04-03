#include "spsc_queue.h"
#include <benchmark/benchmark.h>
#include <thread>
#include <atomic>
#include <vector>

using namespace lockfree;

template <typename Q>
void BM_SPSC(benchmark::State& state) {
    const int count = state.range(0);
    
    static std::atomic<Q*> q_ptr{nullptr};
    static std::atomic<int> ready_count{0};
    static std::atomic<int> finished_count{0};

    if (state.thread_index() == 0) {
        q_ptr.store(new Q());
        ready_count.store(0);
        finished_count.store(0);
    }
    
    ready_count.fetch_add(1);
    while (ready_count.load() < 2) std::this_thread::yield();

    Q* queue = q_ptr.load();

    if (state.thread_index() == 0) {
        for (auto _ : state) {
            for (int i = 0; i < count; ++i) {
                while (!queue->Push(i));
            }
        }
        state.SetItemsProcessed(state.iterations() * count);
    } else {
        for (auto _ : state) {
            for (int i = 0; i < count; ++i) {
                int val;
                while (!queue->Pop(val));
            }
        }
    }

    finished_count.fetch_add(1);
    while (finished_count.load() < 2) std::this_thread::yield();

    if (state.thread_index() == 0) {
        delete queue;
        q_ptr.store(nullptr);
    }
}

#define BENCHMARK_SPSC(QueueType) \
    BENCHMARK_TEMPLATE(BM_SPSC, QueueType<int, 65536>)->Arg(1000000)->Threads(2)->Unit(benchmark::kMillisecond);

BENCHMARK_SPSC(SPSCQueueNaive);
BENCHMARK_SPSC(SPSCQueueAcqRel);
BENCHMARK_SPSC(SPSCQueuePadded);
BENCHMARK_SPSC(SPSCQueueCached);

BENCHMARK_MAIN();
