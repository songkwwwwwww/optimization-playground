#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "object_pool.h"

namespace memory_pool {
namespace {

struct Payload {
  explicit Payload(std::size_t seed) : id(seed) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = seed + i;
    }
  }

  std::size_t id;
  std::array<std::size_t, 8> values;
};

void BM_NewDeleteBatch(benchmark::State& state) {
  const int count = static_cast<int>(state.range(0));
  std::vector<Payload*> objects;
  objects.reserve(count);

  for (auto _ : state) {
    objects.clear();
    for (int i = 0; i < count; ++i) {
      Payload* object = new Payload(static_cast<std::size_t>(i));
      benchmark::DoNotOptimize(object);
      objects.push_back(object);
    }
    for (Payload* object : objects) {
      benchmark::DoNotOptimize(object);
      delete object;
    }
  }

  state.SetItemsProcessed(state.iterations() * count);
}

void BM_ObjectPoolBatch(benchmark::State& state) {
  const int count = static_cast<int>(state.range(0));
  ObjectPool<Payload, 1024> pool;
  std::vector<Payload*> objects;
  objects.reserve(count);

  for (auto _ : state) {
    objects.clear();
    for (int i = 0; i < count; ++i) {
      Payload* object = pool.New(static_cast<std::size_t>(i));
      benchmark::DoNotOptimize(object);
      objects.push_back(object);
    }
    for (Payload* object : objects) {
      benchmark::DoNotOptimize(object);
      pool.Delete(object);
    }
  }

  state.SetItemsProcessed(state.iterations() * count);
}

void BM_NewDeleteInterleaved(benchmark::State& state) {
  for (auto _ : state) {
    Payload* object = new Payload(42);
    benchmark::DoNotOptimize(object);
    delete object;
  }

  state.SetItemsProcessed(state.iterations());
}

void BM_ObjectPoolInterleaved(benchmark::State& state) {
  ObjectPool<Payload, 1024> pool;

  for (auto _ : state) {
    Payload* object = pool.New(42);
    benchmark::DoNotOptimize(object);
    pool.Delete(object);
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_NewDeleteBatch)
    ->Arg(1 << 10)
    ->Arg(1 << 15)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ObjectPoolBatch)
    ->Arg(1 << 10)
    ->Arg(1 << 15)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_NewDeleteInterleaved)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_ObjectPoolInterleaved)->Unit(benchmark::kNanosecond);

}  // namespace
}  // namespace memory_pool

BENCHMARK_MAIN();
