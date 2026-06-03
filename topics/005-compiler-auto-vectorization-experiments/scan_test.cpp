#include "scan.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace autovec_search {
namespace {

TEST(FindByteTest, FindsNeedleInBoundedBuffer) {
  constexpr char kText[] = "compiler-vectorization";

  EXPECT_EQ(FindByte(kText, std::strlen(kText), 'c'), 0);
  EXPECT_EQ(FindByte(kText, std::strlen(kText), 'v'), 9);
  EXPECT_EQ(FindByte(kText, std::strlen(kText), 'n'), 21);
}

TEST(FindByteTest, ReturnsNotFoundWhenNeedleIsAbsent) {
  constexpr char kText[] = "bounded-search";

  EXPECT_EQ(FindByte(kText, std::strlen(kText), 'z'), kNotFound);
}

TEST(FindByteTest, StopsAtFirstEmbeddedNull) {
  constexpr std::array<char, 7> kData = {'a', 'b', '\0', 'c', '\0', 'd', 'e'};

  EXPECT_EQ(FindByte(kData.data(), kData.size(), '\0'), 2);
}

TEST(StrlenTest, ByteByByteAndFakeLengthMatchStdStrlen) {
  const std::vector<std::string> cases = {
      "", "a", "short string", std::string(128, 'x'), std::string(4096, 'y'),
  };

  for (const std::string& text : cases) {
    EXPECT_EQ(ByteByByteStrlen(text.c_str()), std::strlen(text.c_str()));
    EXPECT_EQ(FakeLengthStrlen(text.c_str()), std::strlen(text.c_str()));
  }
}

TEST(FindXorTerminatorTest, FindsFirstMatchingTransformedByte) {
  constexpr std::uint8_t kMask = 0x5a;
  constexpr std::uint8_t kTarget = 0x17;
  std::array<std::uint8_t, 8> data = {1, 2, 3, 4, 5, 6, 7, 8};
  data[5] = kTarget ^ kMask;

  EXPECT_EQ(FindXorTerminator(data.data(), data.size(), kMask, kTarget), 5);
  EXPECT_EQ(ByteByByteFindXorTerminator(data.data(), kMask, kTarget), 5);
  EXPECT_EQ(FakeLengthFindXorTerminator(data.data(), kMask, kTarget), 5);
}

TEST(FindXorTerminatorTest, ReturnsNotFoundForBoundedMiss) {
  constexpr std::uint8_t kMask = 0xa5;
  constexpr std::uint8_t kTarget = 0x00;
  constexpr std::array<std::uint8_t, 4> kData = {1, 2, 3, 4};

  EXPECT_EQ(FindXorTerminator(kData.data(), kData.size(), kMask, kTarget),
            kNotFound);
}

}  // namespace
}  // namespace autovec_search
