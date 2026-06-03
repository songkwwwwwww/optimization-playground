#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace autovec_search {

constexpr std::size_t kNotFound = std::numeric_limits<std::size_t>::max();

#if defined(__has_attribute)
#if __has_attribute(always_inline)
#define AUTOVEC_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define AUTOVEC_ALWAYS_INLINE inline
#endif
#if __has_attribute(noinline)
#define AUTOVEC_NOINLINE __attribute__((noinline))
#else
#define AUTOVEC_NOINLINE
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define AUTOVEC_ALWAYS_INLINE __attribute__((always_inline)) inline
#define AUTOVEC_NOINLINE __attribute__((noinline))
#else
#define AUTOVEC_ALWAYS_INLINE inline
#define AUTOVEC_NOINLINE
#endif

AUTOVEC_ALWAYS_INLINE std::size_t FindByte(const char* haystack,
                                           std::size_t length,
                                           char needle) noexcept {
  for (std::size_t i = 0; i < length; ++i) {
    if (haystack[i] == needle) {
      return i;
    }
  }
  return kNotFound;
}

AUTOVEC_ALWAYS_INLINE std::size_t FindXorTerminator(
    const std::uint8_t* data, std::size_t length, std::uint8_t mask,
    std::uint8_t target) noexcept {
  for (std::size_t i = 0; i < length; ++i) {
    if ((data[i] ^ mask) == target) {
      return i;
    }
  }
  return kNotFound;
}

AUTOVEC_NOINLINE std::size_t ByteByByteStrlen(const char* str) noexcept;
AUTOVEC_NOINLINE std::size_t FakeLengthStrlen(const char* str) noexcept;

AUTOVEC_NOINLINE std::size_t ByteByByteFindXorTerminator(
    const std::uint8_t* data, std::uint8_t mask, std::uint8_t target) noexcept;
AUTOVEC_NOINLINE std::size_t FakeLengthFindXorTerminator(
    const std::uint8_t* data, std::uint8_t mask, std::uint8_t target) noexcept;

#undef AUTOVEC_ALWAYS_INLINE
#undef AUTOVEC_NOINLINE

}  // namespace autovec_search
