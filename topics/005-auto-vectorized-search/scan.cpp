#include "scan.h"

#include <limits>

namespace autovec_search {

std::size_t ByteByByteStrlen(const char* str) noexcept {
  for (std::size_t i = 0;; ++i) {
    if (str[i] == '\0') {
      return i;
    }
  }
}

std::size_t FakeLengthStrlen(const char* str) noexcept {
  return FindByte(str, std::numeric_limits<std::size_t>::max(), '\0');
}

std::size_t ByteByByteFindXorTerminator(const std::uint8_t* data,
                                        std::uint8_t mask,
                                        std::uint8_t target) noexcept {
  for (std::size_t i = 0;; ++i) {
    if ((data[i] ^ mask) == target) {
      return i;
    }
  }
}

std::size_t FakeLengthFindXorTerminator(const std::uint8_t* data,
                                        std::uint8_t mask,
                                        std::uint8_t target) noexcept {
  return FindXorTerminator(data, std::numeric_limits<std::size_t>::max(), mask,
                           target);
}

}  // namespace autovec_search
