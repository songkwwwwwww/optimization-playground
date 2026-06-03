#include "count_even.h"

#include <algorithm>
#include <cstdint>
#include <span>

namespace autovec_count {
namespace {

bool IsEven(std::uint8_t value) noexcept { return (value & 1) == 0; }

template <typename Accumulator>
Accumulator CountIfEvenWithAccumulator(
    std::span<const std::uint8_t> values) noexcept {
  Accumulator count = 0;
  for (std::uint8_t value : values) {
    if (IsEven(value)) {
      ++count;
    }
  }
  return count;
}

}  // namespace

std::ptrdiff_t StdCountIfEven(std::span<const std::uint8_t> values) noexcept {
  return std::count_if(values.begin(), values.end(), IsEven);
}

std::uint8_t CountIfEvenU8(std::span<const std::uint8_t> values) noexcept {
  return CountIfEvenWithAccumulator<std::uint8_t>(values);
}

std::uint16_t CountIfEvenU16(std::span<const std::uint8_t> values) noexcept {
  return CountIfEvenWithAccumulator<std::uint16_t>(values);
}

std::uint32_t CountIfEvenU32(std::span<const std::uint8_t> values) noexcept {
  return CountIfEvenWithAccumulator<std::uint32_t>(values);
}

}  // namespace autovec_count
