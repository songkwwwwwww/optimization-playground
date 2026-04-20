# Optimization Playground (optimization-playground)

Welcome to the **Optimization Playground**. This repository is dedicated to the systematic study and implementation of software performance optimizations. Here, we explore the intersection of machine learning infrastructure, low-level hardware utilization, and concurrent programming.

## 🎯 Project Goals

- **Deep Dive into Modern Hardware:** Understand how CPU caches, SIMD (Single Instruction Multiple Data), and multi-core architectures affect performance.
- **Systematic Optimization:** Go beyond "fast" by using benchmarks and unit tests to quantify and verify every improvement.
- **Reference Implementations:** Build a library of clear, optimized implementations for common algorithms and data structures.

## 🛠 Tech Stack

- **Language:** C++23 (Adhering to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html))
- **Build System:** [Bazel](https://bazel.build/)
- **Benchmarking:** [Google Benchmark](https://github.com/google/benchmark)
- **Testing:** [Google Test](https://github.com/google/googletest)
- **Hardware Abstraction:** ARM NEON Intrinsics (SIMD), OpenMP (Parallelism)

## 📚 Study Logs & Topics

Every experiment is organized as a "topic" within the `topics/` directory. Each topic contains its own documentation, benchmarks, and tests.

### 1. [Matrix Multiplication (GEMM)](topics/001-matmul)
Explores the "purgatory" of matrix multiplication, from naive $O(N^3)$ loops to highly optimized, hardware-aware implementations.
- **Key Techniques:** Loop reordering, Cache tiling (Blocking), Packing, SIMD (NEON), Multi-threading (OpenMP).

### 2. [Lock-free Queue](topics/002-lock-free-queue)
A study on concurrent data structures without traditional mutexes.
- **Key Techniques:** SPSC (Single Producer Single Consumer), CAS (Compare-and-Swap), Memory Barriers, Cache-line alignment.

### 3. [Flat Hash Map](topics/003-flat-hash-map)
A benchmark-driven comparison between `std::unordered_map` and `absl::flat_hash_map`.
- **Key Techniques:** Cache-friendly hash tables, open addressing, pointer chasing, lookup and iteration benchmarks.

## 🚀 Getting Started

### Prerequisites

- **Bazel:** Install via `brew install bazelisk` (macOS) or your preferred package manager.
- **OpenMP:** On macOS, `brew install libomp` is required.
- **C++ Compiler:** Clang or GCC with C++17 support.

### Running Tests

To verify the correctness of all implementations:
```bash
bazel test //...
```

### Running Benchmarks

To measure performance (ensure you build in optimized mode):
```bash
bazel run -c opt //topics/001-matmul:matmul_bench
```

### Updating Compile Commands (for IDEs)

If you use VSCode or CLion with clangd, update the `compile_commands.json`:
```bash
bazel run @hedron_compile_commands//:refresh_compile_commands
```

## 📜 Development Standards

- **Correctness First:** Every optimization must pass correctness tests before being benchmarked.
- **Benchmark-driven:** We don't guess; we measure. All performance claims must be backed by `google_benchmark` results.
- **Clean Code:** Performance should not come at the cost of total unreadability. We follow the Google C++ Style Guide.

## ➕ Adding a New Topic

1. **Create a folder** in `topics/` (e.g., `topics/003-new-topic/`).
2. **Implement your logic** and a corresponding `BUILD` file.
3. **Write a README.md** inside the folder explaining the technique and benchmarking results.
4. **Update the root README.md** to include the new topic in the list.
5. **Update Compile Commands:** `bazel run @hedron_compile_commands//:refresh_compile_commands` to get IDE support for the new files.

---
*Happy Optimizing!*
