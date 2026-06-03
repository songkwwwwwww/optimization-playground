#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "count_even.h"

namespace autovec_count {
namespace {

std::vector<std::uint8_t> MakeMostlyOddValues(std::size_t length,
                                              std::size_t even_count) {
  std::vector<std::uint8_t> values(length, 1);
  even_count = std::min(even_count, values.size());
  for (std::size_t i = 0; i < even_count; ++i) {
    values[i] = static_cast<std::uint8_t>((2 * i) & 0xfe);
  }
  return values;
}

void BM_StdCountIfEven(benchmark::State& state) {
  const std::size_t length = static_cast<std::size_t>(state.range(0));
  const std::vector<std::uint8_t> values =
      MakeMostlyOddValues(length, /*even_count=*/255);

  for (auto _ : state) {
    benchmark::DoNotOptimize(StdCountIfEven(values));
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * length));
}

void BM_CountIfEvenU8(benchmark::State& state) {
  const std::size_t length = static_cast<std::size_t>(state.range(0));
  const std::vector<std::uint8_t> values =
      MakeMostlyOddValues(length, /*even_count=*/255);

  for (auto _ : state) {
    benchmark::DoNotOptimize(CountIfEvenU8(values));
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * length));
}

void BM_CountIfEvenU16(benchmark::State& state) {
  const std::size_t length = static_cast<std::size_t>(state.range(0));
  const std::vector<std::uint8_t> values =
      MakeMostlyOddValues(length, /*even_count=*/255);

  for (auto _ : state) {
    benchmark::DoNotOptimize(CountIfEvenU16(values));
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * length));
}

void BM_CountIfEvenU32(benchmark::State& state) {
  const std::size_t length = static_cast<std::size_t>(state.range(0));
  const std::vector<std::uint8_t> values =
      MakeMostlyOddValues(length, /*even_count=*/255);

  for (auto _ : state) {
    benchmark::DoNotOptimize(CountIfEvenU32(values));
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * length));
}

void ApplyCountSizes(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1024)
      ->Arg(4096)
      ->Arg(16384)
      ->Arg(65536)
      ->Arg(262144)
      ->Arg(1048576)
      ->Arg(4194304);
}

BENCHMARK(BM_StdCountIfEven)->Apply(ApplyCountSizes);
BENCHMARK(BM_CountIfEvenU8)->Apply(ApplyCountSizes);
BENCHMARK(BM_CountIfEvenU16)->Apply(ApplyCountSizes);
BENCHMARK(BM_CountIfEvenU32)->Apply(ApplyCountSizes);

}  // namespace
}  // namespace autovec_count

BENCHMARK_MAIN();
