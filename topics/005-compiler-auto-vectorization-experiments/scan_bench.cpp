#include "scan.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

namespace autovec_search {
namespace {

std::string MakeTerminatedString(std::size_t length) {
  std::string text(length, 'x');
  for (std::size_t i = 0; i < text.size(); ++i) {
    text[i] = static_cast<char>('a' + (i % 23));
  }
  return text;
}

std::vector<std::uint8_t> MakeXorTerminatedData(std::size_t length,
                                                std::uint8_t mask,
                                                std::uint8_t target) {
  std::vector<std::uint8_t> data(length + 1);
  const std::uint8_t terminator = target ^ mask;
  for (std::size_t i = 0; i < length; ++i) {
    std::uint8_t value = static_cast<std::uint8_t>((i * 131 + 17) & 0xff);
    if (value == terminator) {
      value ^= 0x1;
    }
    data[i] = value;
  }
  data[length] = terminator;
  return data;
}

void BM_ByteByByteStrlen(benchmark::State& state) {
  const std::size_t length = static_cast<std::size_t>(state.range(0));
  const std::string text = MakeTerminatedString(length);

  for (auto _ : state) {
    benchmark::DoNotOptimize(ByteByByteStrlen(text.c_str()));
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * length));
}

void BM_FakeLengthStrlen(benchmark::State& state) {
  const std::size_t length = static_cast<std::size_t>(state.range(0));
  const std::string text = MakeTerminatedString(length);

  for (auto _ : state) {
    benchmark::DoNotOptimize(FakeLengthStrlen(text.c_str()));
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * length));
}

void BM_BoundedFindByte(benchmark::State& state) {
  const std::size_t length = static_cast<std::size_t>(state.range(0));
  const std::string text = MakeTerminatedString(length);

  for (auto _ : state) {
    benchmark::DoNotOptimize(FindByte(text.data(), text.size(), '\0'));
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * length));
}

void BM_ByteByByteXorTerminator(benchmark::State& state) {
  constexpr std::uint8_t kMask = 0x5a;
  constexpr std::uint8_t kTarget = 0x17;
  const std::size_t length = static_cast<std::size_t>(state.range(0));
  const std::vector<std::uint8_t> data =
      MakeXorTerminatedData(length, kMask, kTarget);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        ByteByByteFindXorTerminator(data.data(), kMask, kTarget));
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * length));
}

void BM_FakeLengthXorTerminator(benchmark::State& state) {
  constexpr std::uint8_t kMask = 0x5a;
  constexpr std::uint8_t kTarget = 0x17;
  const std::size_t length = static_cast<std::size_t>(state.range(0));
  const std::vector<std::uint8_t> data =
      MakeXorTerminatedData(length, kMask, kTarget);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        FakeLengthFindXorTerminator(data.data(), kMask, kTarget));
  }

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * length));
}

void ApplyScanSizes(benchmark::Benchmark* benchmark) {
  benchmark->Arg(64)->Arg(1024)->Arg(65536)->Arg(1048576);
}

BENCHMARK(BM_ByteByByteStrlen)->Apply(ApplyScanSizes);
BENCHMARK(BM_FakeLengthStrlen)->Apply(ApplyScanSizes);
BENCHMARK(BM_BoundedFindByte)->Apply(ApplyScanSizes);
BENCHMARK(BM_ByteByByteXorTerminator)->Apply(ApplyScanSizes);
BENCHMARK(BM_FakeLengthXorTerminator)->Apply(ApplyScanSizes);

}  // namespace
}  // namespace autovec_search

BENCHMARK_MAIN();
