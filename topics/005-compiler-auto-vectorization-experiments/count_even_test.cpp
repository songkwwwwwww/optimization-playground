#include "count_even.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace autovec_count {
namespace {

std::vector<std::uint8_t> MakeValuesWithEvenCount(std::size_t length,
                                                  std::size_t even_count) {
  std::vector<std::uint8_t> values(length, 1);
  for (std::size_t i = 0; i < even_count; ++i) {
    values[i] = 2;
  }
  return values;
}

TEST(CountEvenTest, EmptyInputReturnsZero) {
  const std::vector<std::uint8_t> values;

  EXPECT_EQ(StdCountIfEven(values), 0);
  EXPECT_EQ(CountIfEvenU8(values), 0);
  EXPECT_EQ(CountIfEvenU16(values), 0);
  EXPECT_EQ(CountIfEvenU32(values), 0);
}

TEST(CountEvenTest, CountsMixedValues) {
  const std::vector<std::uint8_t> values = {0, 1, 2, 3, 4, 5, 6, 7};

  EXPECT_EQ(StdCountIfEven(values), 4);
  EXPECT_EQ(CountIfEvenU8(values), 4);
  EXPECT_EQ(CountIfEvenU16(values), 4);
  EXPECT_EQ(CountIfEvenU32(values), 4);
}

TEST(CountEvenTest, SupportsU8BoundedResult) {
  const std::vector<std::uint8_t> values =
      MakeValuesWithEvenCount(/*length=*/4096, /*even_count=*/255);

  EXPECT_EQ(StdCountIfEven(values), 255);
  EXPECT_EQ(CountIfEvenU8(values), 255);
  EXPECT_EQ(CountIfEvenU16(values), 255);
  EXPECT_EQ(CountIfEvenU32(values), 255);
}

TEST(CountEvenTest, WiderAccumulatorsSupportLargerKnownRanges) {
  const std::vector<std::uint8_t> values =
      MakeValuesWithEvenCount(/*length=*/4096, /*even_count=*/1024);

  EXPECT_EQ(StdCountIfEven(values), 1024);
  EXPECT_EQ(CountIfEvenU16(values), 1024);
  EXPECT_EQ(CountIfEvenU32(values), 1024);
}

TEST(CountEvenTest, U8AccumulatorIsModuloWhenPreconditionIsViolated) {
  const std::vector<std::uint8_t> values =
      MakeValuesWithEvenCount(/*length=*/4096, /*even_count=*/256);

  EXPECT_EQ(StdCountIfEven(values), 256);
  EXPECT_EQ(CountIfEvenU8(values), 0);
}

}  // namespace
}  // namespace autovec_count
