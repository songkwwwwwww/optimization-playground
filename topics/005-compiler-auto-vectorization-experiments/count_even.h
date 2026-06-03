#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace autovec_count {

#if defined(__has_attribute)
#if __has_attribute(noinline)
#define AUTOVEC_COUNT_NOINLINE __attribute__((noinline))
#else
#define AUTOVEC_COUNT_NOINLINE
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define AUTOVEC_COUNT_NOINLINE __attribute__((noinline))
#else
#define AUTOVEC_COUNT_NOINLINE
#endif

AUTOVEC_COUNT_NOINLINE std::ptrdiff_t StdCountIfEven(
    std::span<const std::uint8_t> values) noexcept;

AUTOVEC_COUNT_NOINLINE std::uint8_t CountIfEvenU8(
    std::span<const std::uint8_t> values) noexcept;

AUTOVEC_COUNT_NOINLINE std::uint16_t CountIfEvenU16(
    std::span<const std::uint8_t> values) noexcept;

AUTOVEC_COUNT_NOINLINE std::uint32_t CountIfEvenU32(
    std::span<const std::uint8_t> values) noexcept;

#undef AUTOVEC_COUNT_NOINLINE

}  // namespace autovec_count
