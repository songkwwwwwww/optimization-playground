#ifndef TOPICS_003_FLAT_HASH_MAP_HASH_TABLE_WORKLOADS_H_
#define TOPICS_003_FLAT_HASH_MAP_HASH_TABLE_WORKLOADS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hash_table_lab {

struct Entry {
  std::uint64_t key;
  std::uint64_t value;
};

inline std::uint64_t Mix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

inline std::vector<Entry> MakeSequentialEntries(std::size_t count) {
  std::vector<Entry> entries;
  entries.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint64_t key = static_cast<std::uint64_t>(i);
    entries.push_back({key, Mix64(key)});
  }
  return entries;
}

inline std::vector<Entry> MakeRandomizedEntries(std::size_t count) {
  constexpr std::uint64_t kKeySeed = 0x100000000ULL;
  constexpr std::uint64_t kValueSeed = 0x517cc1b727220a95ULL;

  std::vector<Entry> entries;
  entries.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint64_t key = Mix64(kKeySeed + static_cast<std::uint64_t>(i));
    entries.push_back({key, Mix64(key ^ kValueSeed)});
  }
  return entries;
}

inline std::vector<std::uint64_t> MakeHitQueries(
    const std::vector<Entry>& entries, std::size_t count) {
  std::vector<std::uint64_t> queries;
  queries.reserve(count);
  if (entries.empty()) {
    return queries;
  }

  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t index = Mix64(static_cast<std::uint64_t>(i)) %
                              static_cast<std::uint64_t>(entries.size());
    queries.push_back(entries[index].key);
  }
  return queries;
}

inline std::vector<std::uint64_t> MakeMissQueries(std::size_t count) {
  constexpr std::uint64_t kMissingKeySeed = 0x8000000000000000ULL;

  std::vector<std::uint64_t> queries;
  queries.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    queries.push_back(Mix64(kMissingKeySeed + static_cast<std::uint64_t>(i)));
  }
  return queries;
}

template <typename Map>
Map BuildMap(const std::vector<Entry>& entries) {
  Map map;
  map.reserve(entries.size());
  for (const Entry& entry : entries) {
    map.emplace(entry.key, entry.value);
  }
  return map;
}

template <typename Map>
std::uint64_t SumSuccessfulLookups(
    const Map& map, const std::vector<std::uint64_t>& queries) {
  std::uint64_t sum = 0;
  for (const std::uint64_t key : queries) {
    const auto it = map.find(key);
    if (it != map.end()) {
      sum += it->second;
    }
  }
  return sum;
}

template <typename Map>
std::size_t CountMissingLookups(
    const Map& map, const std::vector<std::uint64_t>& queries) {
  std::size_t misses = 0;
  for (const std::uint64_t key : queries) {
    if (map.find(key) == map.end()) {
      ++misses;
    }
  }
  return misses;
}

template <typename Map>
std::uint64_t SumIterationValues(const Map& map) {
  std::uint64_t sum = 0;
  for (const auto& [key, value] : map) {
    sum += Mix64(key) ^ value;
  }
  return sum;
}

}  // namespace hash_table_lab

#endif  // TOPICS_003_FLAT_HASH_MAP_HASH_TABLE_WORKLOADS_H_
