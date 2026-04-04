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
| **Naive** | ~185 M/s | Efficient on ARM64 due to native load-acquire/store-release. |
| **Acquire-Release** | ~186 M/s | Minimal overhead difference from Naive on this architecture. |
| **Padded** | ~200 M/s | **Fastest.** Eliminates false sharing between indices. |
| **Cached** | ~35 M/s | **Performance Regression.** 5-6x slower due to branch overhead. |

### Analysis of "Cached" Performance
The "Cached" optimization (Stage 4) is intended to reduce cross-core cache coherence traffic. However, on Apple Silicon:
- **Shared L2/L3 Cache**: Cores within a cluster share a fast L2 cache, making `atomic load` very cheap.
- **Branch vs. Atomic**: The overhead of additional branching (`if (cache == value)`) and local variable updates outweighs the cost of a direct atomic load from the shared cache.
- **Pipeline Stalls**: Extra logic can introduce data dependencies that hinder the CPU's out-of-order execution engine.

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
