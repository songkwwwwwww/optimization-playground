#include "hash_table_workloads.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "gtest/gtest.h"

namespace hash_table_lab {
namespace {

using FlatHashMap = absl::flat_hash_map<std::uint64_t, std::uint64_t>;
using StdUnorderedMap = std::unordered_map<std::uint64_t, std::uint64_t>;

TEST(HashTableWorkloadTest, RandomizedEntriesHaveUniqueKeys) {
  const std::vector<Entry> entries = MakeRandomizedEntries(4096);
  std::unordered_set<std::uint64_t> keys;
  keys.reserve(entries.size());

  for (const Entry& entry : entries) {
    EXPECT_TRUE(keys.insert(entry.key).second);
  }
}

TEST(HashTableWorkloadTest, HitQueriesComeFromInputEntries) {
  const std::vector<Entry> entries = MakeRandomizedEntries(256);
  const FlatHashMap map = BuildMap<FlatHashMap>(entries);
  const std::vector<std::uint64_t> queries = MakeHitQueries(entries, 4096);

  ASSERT_EQ(queries.size(), 4096);
  for (const std::uint64_t key : queries) {
    EXPECT_NE(map.find(key), map.end());
  }
}

TEST(HashTableWorkloadTest, MissQueriesDoNotHitInputEntries) {
  const std::vector<Entry> entries = MakeRandomizedEntries(4096);
  const FlatHashMap map = BuildMap<FlatHashMap>(entries);
  const std::vector<std::uint64_t> queries = MakeMissQueries(4096);

  EXPECT_EQ(CountMissingLookups(map, queries), queries.size());
}

TEST(HashTableComparisonTest, BuildsEquivalentMapsForSequentialKeys) {
  const std::vector<Entry> entries = MakeSequentialEntries(4096);
  const StdUnorderedMap std_map = BuildMap<StdUnorderedMap>(entries);
  const FlatHashMap flat_map = BuildMap<FlatHashMap>(entries);

  ASSERT_EQ(std_map.size(), flat_map.size());
  for (const Entry& entry : entries) {
    const auto std_it = std_map.find(entry.key);
    const auto flat_it = flat_map.find(entry.key);

    ASSERT_NE(std_it, std_map.end());
    ASSERT_NE(flat_it, flat_map.end());
    EXPECT_EQ(std_it->second, entry.value);
    EXPECT_EQ(flat_it->second, entry.value);
  }
}

TEST(HashTableComparisonTest, BuildsEquivalentMapsForRandomizedKeys) {
  const std::vector<Entry> entries = MakeRandomizedEntries(4096);
  const StdUnorderedMap std_map = BuildMap<StdUnorderedMap>(entries);
  const FlatHashMap flat_map = BuildMap<FlatHashMap>(entries);

  ASSERT_EQ(std_map.size(), flat_map.size());
  for (const Entry& entry : entries) {
    const auto std_it = std_map.find(entry.key);
    const auto flat_it = flat_map.find(entry.key);

    ASSERT_NE(std_it, std_map.end());
    ASSERT_NE(flat_it, flat_map.end());
    EXPECT_EQ(std_it->second, flat_it->second);
  }
}

TEST(HashTableComparisonTest, LookupHelpersAgreeAcrossMapTypes) {
  const std::vector<Entry> entries = MakeRandomizedEntries(4096);
  const std::vector<std::uint64_t> hit_queries = MakeHitQueries(entries, 8192);
  const std::vector<std::uint64_t> miss_queries = MakeMissQueries(8192);
  const StdUnorderedMap std_map = BuildMap<StdUnorderedMap>(entries);
  const FlatHashMap flat_map = BuildMap<FlatHashMap>(entries);

  EXPECT_EQ(SumSuccessfulLookups(std_map, hit_queries),
            SumSuccessfulLookups(flat_map, hit_queries));
  EXPECT_EQ(CountMissingLookups(std_map, miss_queries),
            CountMissingLookups(flat_map, miss_queries));
  EXPECT_EQ(SumIterationValues(std_map), SumIterationValues(flat_map));
}

}  // namespace
}  // namespace hash_table_lab
