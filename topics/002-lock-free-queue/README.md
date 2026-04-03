# SPSC Queue Optimization Study

This study explores the performance of a Single Producer Single Consumer (SPSC) Bounded Queue, progressing from a naive implementation to highly optimized versions discussed in the talk "Beyond Sequential Consistency" by Chris Frretz (CppCon).

## Implementations

1.  **Naive**: Uses `std::atomic` with default `memory_order_seq_cst` and no padding. `head_` and `tail_` likely reside in the same cache line (False Sharing).
2.  **Acquire-Release**: Uses `memory_order_acquire` and `memory_order_release` to reduce memory fence overhead, but still no padding.
3.  **Padded**: Adds `alignas(64)` to `head_` and `tail_` to eliminate false sharing.
4.  **Cached**: Further optimization using local snapshots of the "other" index to reduce cache coherence traffic across cores.

## Benchmark Results (Apple Silicon M1/M2/M3)

Run with `bazel run -c opt //topics/002-lock-free-queue:spsc_queue_bench`.

| Implementation | Throughput (items/sec) | Note |
| :--- | :--- | :--- |
| **Naive** | ~178 M/s | Surprisingly fast on ARM64. |
| **Acquire-Release** | ~164 M/s | Slightly slower than Naive? (Likely due to M-series optimization). |
| **Padded** | ~189 M/s | **Fastest.** Padding helps by separating the write-heavy indexes. |
| **Cached** | ~22 M/s | **Surprisingly slow.** Why? |

### Analysis of "Cached" Performance
The "Cached" optimization is intended to reduce Last Level Cache (LLC) misses by avoiding reads of the atomic index owned by the other thread. However, on Apple Silicon:
- The Unified Memory Architecture and large shared L2 cache within a cluster may make cache coherence traffic very cheap.
- The overhead of maintaining `head_cache_` and `tail_cache_` might exceed the benefit if the threads are running on the same cluster.
- If the producer and consumer are extremely fast and the queue frequently hits "full" or "empty" states, the cached version effectively falls back to the padded version but with more branch overhead.

## How to Run

### Run Tests
```bash
bazel test //topics/002-lock-free-queue:spsc_queue_test
```

### Run Benchmarks
```bash
bazel run -c opt //topics/002-lock-free-queue:spsc_queue_bench
```

## References
- [SPSC Optimization Guide](docs/SPSC_OPTIMIZATION.md)
- Chris Frretz, "Beyond Sequential Consistency", CppCon.
- "A Fast Lock-Free Queue for Shared-Memory Multiprocessors", 2009.
