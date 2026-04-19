#include "hash_table_workloads.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "benchmark/benchmark.h"

namespace hash_table_lab {
namespace {

using FlatHashMap = absl::flat_hash_map<std::uint64_t, std::uint64_t>;
using StdUnorderedMap = std::unordered_map<std::uint64_t, std::uint64_t>;

template <typename Map>
void BenchmarkInsertSequential(benchmark::State& state) {
  const std::size_t count = static_cast<std::size_t>(state.range(0));
  const std::vector<Entry> entries = MakeSequentialEntries(count);

  for (auto _ : state) {
    Map map;
    map.reserve(entries.size());
    for (const Entry& entry : entries) {
      benchmark::DoNotOptimize(map.emplace(entry.key, entry.value));
    }
    benchmark::DoNotOptimize(map);
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
}

template <typename Map>
void BenchmarkInsertRandomized(benchmark::State& state) {
  const std::size_t count = static_cast<std::size_t>(state.range(0));
  const std::vector<Entry> entries = MakeRandomizedEntries(count);

  for (auto _ : state) {
    Map map;
    map.reserve(entries.size());
    for (const Entry& entry : entries) {
      benchmark::DoNotOptimize(map.emplace(entry.key, entry.value));
    }
    benchmark::DoNotOptimize(map);
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
}

template <typename Map>
void BenchmarkSuccessfulLookup(benchmark::State& state) {
  const std::size_t count = static_cast<std::size_t>(state.range(0));
  const std::vector<Entry> entries = MakeRandomizedEntries(count);
  const std::vector<std::uint64_t> queries = MakeHitQueries(entries, count);
  const Map map = BuildMap<Map>(entries);

  for (auto _ : state) {
    std::uint64_t sum = SumSuccessfulLookups(map, queries);
    benchmark::DoNotOptimize(sum);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
}

template <typename Map>
void BenchmarkMissingLookup(benchmark::State& state) {
  const std::size_t count = static_cast<std::size_t>(state.range(0));
  const std::vector<Entry> entries = MakeRandomizedEntries(count);
  const std::vector<std::uint64_t> queries = MakeMissQueries(count);
  const Map map = BuildMap<Map>(entries);

  for (auto _ : state) {
    std::size_t misses = CountMissingLookups(map, queries);
    benchmark::DoNotOptimize(misses);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
}

template <typename Map>
void BenchmarkIteration(benchmark::State& state) {
  const std::size_t count = static_cast<std::size_t>(state.range(0));
  const std::vector<Entry> entries = MakeRandomizedEntries(count);
  const Map map = BuildMap<Map>(entries);

  for (auto _ : state) {
    std::uint64_t sum = SumIterationValues(map);
    benchmark::DoNotOptimize(sum);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
}

#define REGISTER_MAP_BENCHMARK(BenchmarkName, MapType, MapLabel)          \
  BENCHMARK_TEMPLATE(BenchmarkName, MapType)                              \
      ->RangeMultiplier(4)                                                \
      ->Range(1 << 10, 1 << 18)                                           \
      ->Unit(benchmark::kMicrosecond)                                     \
      ->Name(#BenchmarkName "/" MapLabel)

REGISTER_MAP_BENCHMARK(BenchmarkInsertSequential, StdUnorderedMap,
                       "std_unordered_map");
REGISTER_MAP_BENCHMARK(BenchmarkInsertSequential, FlatHashMap,
                       "absl_flat_hash_map");

REGISTER_MAP_BENCHMARK(BenchmarkInsertRandomized, StdUnorderedMap,
                       "std_unordered_map");
REGISTER_MAP_BENCHMARK(BenchmarkInsertRandomized, FlatHashMap,
                       "absl_flat_hash_map");

REGISTER_MAP_BENCHMARK(BenchmarkSuccessfulLookup, StdUnorderedMap,
                       "std_unordered_map");
REGISTER_MAP_BENCHMARK(BenchmarkSuccessfulLookup, FlatHashMap,
                       "absl_flat_hash_map");

REGISTER_MAP_BENCHMARK(BenchmarkMissingLookup, StdUnorderedMap,
                       "std_unordered_map");
REGISTER_MAP_BENCHMARK(BenchmarkMissingLookup, FlatHashMap,
                       "absl_flat_hash_map");

REGISTER_MAP_BENCHMARK(BenchmarkIteration, StdUnorderedMap,
                       "std_unordered_map");
REGISTER_MAP_BENCHMARK(BenchmarkIteration, FlatHashMap,
                       "absl_flat_hash_map");

}  // namespace
}  // namespace hash_table_lab

BENCHMARK_MAIN();
