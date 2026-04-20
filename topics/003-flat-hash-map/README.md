# 003: Flat Hash Map

This topic compares `std::unordered_map` with `absl::flat_hash_map` on small
`uint64_t -> uint64_t` workloads.

The goal is not to prove that one container is always better. The goal is to
make the memory-layout trade-off visible:

- `std::unordered_map` is commonly node-based. Each element may live in a
  separately allocated node, so lookup often involves pointer chasing.
- `absl::flat_hash_map` stores entries in a flat, open-addressed table. This can
  improve locality, but it also changes iterator stability and rehash behavior.

## Workloads

The benchmark covers five basic operations:

1. Sequential-key insertion with `reserve`.
2. Randomized-key insertion with `reserve`.
3. Successful lookup on randomized keys.
4. Missing lookup on randomized keys.
5. Full-table iteration.

The input generation is deterministic. `hash_table_workloads.h` uses a small
SplitMix64-style mixer to create reproducible keys and values without measuring
random-number generation inside the benchmark loop.

## How to Run

Run correctness tests:

```bash
bazel test //topics/003-flat-hash-map:flat_hash_map_test
```

Run all benchmarks:

```bash
bazel run -c opt //topics/003-flat-hash-map:flat_hash_map_bench
```

Run only lookup benchmarks:

```bash
bazel run -c opt //topics/003-flat-hash-map:flat_hash_map_bench -- --benchmark_filter="Lookup"
```

Run one map implementation:

```bash
bazel run -c opt //topics/003-flat-hash-map:flat_hash_map_bench -- --benchmark_filter="absl_flat_hash_map"
```

## What to Look For

Useful questions while reading the benchmark output:

- Does the flat table win more clearly on lookup or iteration?
- How much does randomized insertion differ from sequential insertion?
- Does the result change as the table grows beyond cache-friendly sizes?
- Is `reserve` enough to remove most rehash noise from the insert benchmark?
- Which behavior matters more for the workload: lookup latency, iteration
  throughput, memory overhead, or pointer/iterator stability?

## Initial Sanity Run

The table below is a short local run for successful lookups only. Treat it as a
sanity check, not a final result. Full benchmark results should be collected on
an otherwise quiet machine with a longer `--benchmark_min_time`.

Command:

```bash
bazel run -c opt //topics/003-flat-hash-map:flat_hash_map_bench -- --benchmark_filter="SuccessfulLookup" --benchmark_min_time=0.01s
```

| Entries | `std::unordered_map` | `absl::flat_hash_map` |
| --- | ---: | ---: |
| 1,024 | 959.241M items/s | 1.05025G items/s |
| 4,096 | 930.341M items/s | 928.475M items/s |
| 16,384 | 588.837M items/s | 826.164M items/s |
| 65,536 | 269.469M items/s | 711.034M items/s |
| 262,144 | 174.015M items/s | 640.268M items/s |

In this run, the flat table starts pulling away once the working set grows.
That is the behavior this topic is meant to investigate more carefully:
contiguous probing often gives the CPU friendlier memory access than following
separate heap nodes.

## Notes

This first version intentionally uses small integer keys and values. Follow-up
experiments can add:

- larger value payloads,
- string keys,
- erase-heavy workloads,
- mixed insert/find/update traces,
- memory usage counters or allocator instrumentation.

## References

- Abseil `flat_hash_map`: <https://abseil.io/docs/cpp/guides/container>
- Abseil Swiss Tables design notes: <https://abseil.io/about/design/swisstables>
- CppCon 2017, Matt Kulukundis, Designing a Fast, Efficient, Cache-friendly Hash Table: <https://www.youtube.com/watch?v=ncHmEUmJZf4>
