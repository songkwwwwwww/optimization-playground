# 001: Matrix Multiplication (Matmul)

The goal of this project is not to write a competitive BLAS implementation, but to learn about common performance optimizations

This directory contains various implementations of the Matrix Multiplication (GEMM) algorithm, progressing from a naive approach to highly optimized hardware-aware versions. The goal is to study how different low-level software optimization techniques affect computational throughput (GFLOPS).

For readability, the study implementations now assume square inputs. The
tile-based variants also assume the matrix size is divisible by their tile
size, so the teaching code can avoid remainder-handling branches.

## Implementations

1. **`matmul_naive.cpp`**: The standard $O(N^3)$ triple-nested loop implementation without any optimizations.
2. **`matmul_naive_register_acc.cpp`**: Keeps the basic loop nest but accumulates each output element in a register before storing it back to memory.
3. **`matmul_loop_reorder.cpp`**: Optimizes memory access patterns (spatial locality) by reordering loops from `row-col-inner` to `row-inner-col`.
4. **`matmul_tiled_1d.cpp`**: Adds a simple 1D tile/blocking strategy to improve cache reuse while keeping the control flow approachable.
5. **`matmul_tiled_md.cpp`**: Extends tiling across multiple dimensions for a more realistic cache-blocked study variant.
6. **`matmul_packed.cpp`**: Packs matrix tiles into continuous memory buffers to minimize TLB (Translation Lookaside Buffer) misses and cache conflicts.
7. **`matmul_simd.cpp`**: Utilizes ARM NEON Intrinsics to compute multiple data points in a single instruction cycle (Vectorization).
8. **`matmul_omp.cpp` / `OmpThread`**: A cache-tiled OpenMP baseline specialized for the fixed `2048 x 2048 x 2048` study workload, so the code can stay focused on the threading idea.
9. **`matmul_omp.cpp` / `OmpThreadSimd`**: Adds an explicit SIMD micro-kernel inside each OpenMP tile while keeping the same fixed-size study assumption.
10. **`matmul_omp.cpp` / `OmpThreadPacked`**: Packs per-thread `A`/`B` tiles into contiguous scratch buffers before multiplying them to reduce strided memory traffic.
11. **`matmul_omp.cpp` / `OmpThreadPackedSimd`**: Combines OpenMP tiling, per-thread packing, and SIMD in the micro-kernel to study the stacked effect of all three optimizations.
12. **`matmul_omp.cpp` / `OmpThreadPackedRow`**: Uses the packed SIMD tile kernel but parallelizes only across row blocks to compare scheduling behavior against the `collapse(2)` variants.
13. **`matmul_omp.cpp` / `OmpThreadPackedRegister`**: Uses a packed `4x8` register-blocked micro-kernel so each thread accumulates a small `C` tile in NEON registers before writing it back.
14. **`matmul_reference.cpp`**: Wraps the highly-optimized system BLAS library (e.g., Apple Accelerate Framework or OpenBLAS) for ground-truth correctness and baseline performance comparisons.

## Prerequisites

### Build Tools

- `bazel` or `bazelisk`
- A C++17-compatible compiler (`clang` or `gcc`)

### System Libraries

- OpenMP runtime
  - This project links against `libomp`.
  - On Apple Silicon macOS, the current repository setup expects Homebrew's
    `libomp` at `/opt/homebrew/opt/libomp`.
  - Example:
    ```bash
    brew install libomp
    ```
- BLAS implementation for the reference path
  - macOS: Apple Accelerate framework is used automatically.
  - Non-macOS: OpenBLAS is expected via `-lopenblas`.

### Bazel-managed Dependencies

- `googletest` for correctness tests
- `google_benchmark` for performance benchmarks

These are declared in the repository's `MODULE.bazel`, so they are fetched by
Bazel automatically.

## How to Run

### Run Correctness Tests
Validates that all currently wired custom algorithms compute the correct matrix products against the reference implementation.
```bash
bazel test //topics/001-matmul:matmul_test
```
The OpenMP variants are intentionally checked on the fixed `2048 x 2048`
study workload because their code path now drops ragged-tile handling for
readability.

### Run Performance Benchmarks
Measures the GFLOPS (Giga-Floating Point Operations Per Second) of each method using Google Benchmark.
```bash
bazel run -c opt //topics/001-matmul:matmul_bench
```
The benchmark target now uses one fixed `2048 x 2048` input size so the OpenMP
study variants and the simpler teaching code stay aligned.

To filter and benchmark a specific algorithm:
```bash
bazel run -c opt //topics/001-matmul:matmul_bench -- --benchmark_filter="OmpThread"
```

To compare the entire OpenMP variant family:
```bash
bazel run -c opt //topics/001-matmul:matmul_bench -- --benchmark_filter="OmpThread.*"
```

### Benchmark Results

The benchmark results below were collected on an Apple M4 Mac mini.
Performance numbers are hardware-, compiler-, and build-option-dependent, so
results will vary on other machines. The sample output also predates some implementation
renames and additions, so treat it as illustrative rather than exhaustive.

```
Run on (10 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x10)
Load Average: 1.79, 1.64, 1.66
-------------------------------------------------------------------------------------------------------
Benchmark                                             Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
Benchmark                                             Time             CPU   Iterations UserCounters...
-------------------------------------------------------------------------------------------------------
BenchmarkMatmul/Naive/2048                        20758 ms        20758 ms            1 GFLOPS=827.644M/s
BenchmarkMatmul/NaiveRegisterAcc/2048             16552 ms        16552 ms            1 GFLOPS=1.03796G/s
BenchmarkMatmul/LoopReorder/2048                   1042 ms         1042 ms            1 GFLOPS=16.4941G/s
BenchmarkMatmul/Tiled1D/2048                       1039 ms         1039 ms            1 GFLOPS=16.5426G/s
BenchmarkMatmul/TiledMD/2048                       1982 ms         1982 ms            1 GFLOPS=8.66827G/s
BenchmarkMatmul/SIMD/2048                          1358 ms         1358 ms            1 GFLOPS=12.6491G/s
BenchmarkMatmul/Packed/2048                         721 ms          721 ms            1 GFLOPS=23.8339G/s
BenchmarkMatmul/OmpThread/2048                      261 ms          248 ms            3 GFLOPS=69.1824G/s
BenchmarkMatmul/OmpThreadSimd/2048                  298 ms          268 ms            3 GFLOPS=64.0204G/s
BenchmarkMatmul/OmpThreadPacked/2048                177 ms          167 ms            4 GFLOPS=102.585G/s
BenchmarkMatmul/OmpThreadPackedSimd/2048            220 ms          213 ms            3 GFLOPS=80.6805G/s
BenchmarkMatmul/OmpThreadPackedRow/2048             246 ms          237 ms            3 GFLOPS=72.3483G/s
BenchmarkMatmul/OmpThreadPackedRegister/2048       99.9 ms         83.9 ms            7 GFLOPS=204.828G/s
BenchmarkMatmul/Reference/2048                     42.2 ms         42.2 ms           17 GFLOPS=406.996G/s
```

## References

- [Fast Multidimensional Matrix Multiplication on CPU from Scratch, Simon Boehm, 202208](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Optimizing matrix multiplication - Discovering optimizations one at a time, Michal Pitr, 20250216](https://michalpitr.substack.com/p/optimizing-matrix-multiplication)
- [Matrix Multiplication Deep Dive || Cache Blocking, SIMD & Parallelization - Aliaksei Sala - CppCon2025](https://www.youtube.com/watch?v=GHctcSBd6Z4)
- [MIT’s 6.172 on OCW](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/pages/syllabus/)
- [github.com/flame/how-to-optimize-gemm](https://github.com/flame/how-to-optimize-gemm)
