# SPSC Queue Optimization Study

This study explores the performance of a Single Producer Single Consumer (SPSC) Bounded Queue, progressing from an intentionally broken implementation to highly optimized versions discussed in the talk "Beyond Sequential Consistency" by Chris Fretz (CppCon).

## Implementations

0.  **Broken (Relaxed-Only)**: All atomic operations use `memory_order_relaxed`. No happens-before guarantee — the consumer may observe stale or torn data. Exists solely to demonstrate data races with ThreadSanitizer.
1.  **Naive**: Uses `std::atomic` with default `memory_order_seq_cst` and no padding. `head_` and `tail_` likely reside in the same cache line (False Sharing).
2.  **Acquire-Release**: Uses `memory_order_acquire` and `memory_order_release` to reduce memory fence overhead, but still no padding.
3.  **Padded**: Adds `alignas(64)` to `head_` and `tail_` to eliminate false sharing.
4.  **Cached**: Further optimization using local snapshots of the "other" index to reduce cache coherence traffic across cores.

## Benchmark Results (Apple Silicon M1/M2/M3)

Run with `bazel run -c opt //topics/002-lock-free-queue:spsc_queue_bench`.

### Throughput

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

### x86 (Intel/AMD) — Expected Behavior

On x86-64 with split L2 caches (e.g., dual-socket Xeon, or even desktop parts with non-shared L2):
- **Naive vs. Acquire-Release**: Should show a measurable gap. x86 has Total Store Order (TSO), so `seq_cst` stores emit an `MFENCE` or `LOCK`-prefixed instruction, while `release` stores are plain `MOV`s (free on TSO). Expect ~10-30% improvement.
- **Padded**: Eliminates false sharing — similar benefit as ARM64.
- **Cached**: Should show a **significant speedup** (2-5x) over Padded on multi-socket systems where cross-socket atomic loads traverse QPI/UPI, costing 100+ ns. The local cache avoids this in the common (non-full/non-empty) case.

| Optimization | ARM64 (Apple Silicon) | x86-64 (Split L2) | Why the difference |
| :--- | :--- | :--- | :--- |
| SeqCst → AcqRel | ~0% gain | ~10-30% gain | ARM has native acq/rel; x86 TSO makes seq_cst stores expensive |
| + Padding | ~8% gain | ~10-20% gain | False sharing elimination — similar on both |
| + Caching | **~82% regression** | **~2-5x speedup** | Shared L2 vs. cross-socket QPI/UPI latency difference |

## How to Run

### Run Tests
```bash
bazel test //topics/002-lock-free-queue:spsc_queue_test
```

### Run Benchmarks (Throughput + Latency)
```bash
bazel run -c opt //topics/002-lock-free-queue:spsc_queue_bench
```

The latency benchmark reports p50, p99, and p99.9 per-item latencies (ns) for Padded vs Cached queues.

### Run with ThreadSanitizer (detect data races in the Broken version)
```bash
bazel test --config=tsan //topics/002-lock-free-queue:spsc_queue_test
```

## References
- [SPSC Optimization Guide](docs/SPSC_OPTIMIZATION.md)
- Chris Fretz, "Beyond Sequential Consistency", CppCon.
- "A Fast Lock-Free Queue for Shared-Memory Multiprocessors", 2009.
