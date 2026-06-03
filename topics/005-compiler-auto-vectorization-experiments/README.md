# 005: Compiler Auto-vectorization Experiments

This topic collects experiments that shape ordinary C++ loops so the compiler
has a better chance of emitting SIMD code.

## Experiment 1: Search With a Fake Length

The core question is: can we make a loop that normally has no known trip count,
such as a `strlen`-style scan, look like a bounded buffer search so the compiler
is more willing to auto-vectorize it?

### Idea

The ordinary `strlen` loop has no explicit upper bound:

```cpp
for (std::size_t i = 0;; ++i) {
  if (str[i] == '\0') return i;
}
```

The experimental version reuses a bounded search helper:

```cpp
for (std::size_t i = 0; i < len; ++i) {
  if (haystack[i] == needle) return i;
}
return npos;
```

For a valid null-terminated string, passing `std::numeric_limits<size_t>::max()`
as `len` should behave like an effectively unreachable bound. The loop still
has a syntactic trip count, which can trigger better auto-vectorization on
compiler versions that optimize known-length early-exit scans.

### Implementations

- `ByteByByteStrlen`: direct unknown-length `strlen`-style loop.
- `FakeLengthStrlen`: calls the bounded byte finder with `SIZE_MAX` and
  `'\0'`.
- `FindByte`: ordinary bounded byte search, useful as the compiler-friendly
  building block.
- `FindXorTerminator`: a slightly more complex scan where the stop condition is
  `(data[i] ^ mask) == target`.
- `FakeLengthFindXorTerminator`: the same fake-length trick applied to the XOR
  scan.

The build target uses `-fno-builtin` so the compiler does not replace the
teaching implementation with libc's `strlen`.

## Experiment 2: Count With a Narrow Accumulator

The second experiment studies an idea from the article "Improving on
`std::count_if()`'s auto-vectorization".

The core question is: if we know the result range of a reduction in advance,
can we make the accumulator type narrower so the compiler packs more partial
counts into each SIMD register?

### Idea

`std::count_if` returns the iterator `difference_type`, which is usually a
wide signed integer such as `std::ptrdiff_t`. For a `uint8_t` input buffer, a
compiler may therefore widen each per-element predicate result into 64-bit
lanes before accumulating:

```cpp
auto count = std::count_if(values.begin(), values.end(),
                           [](std::uint8_t x) { return x % 2 == 0; });
```

If the caller already knows the answer is in `[0, 255]`, the accumulator does
not need 64-bit precision. A custom counting loop can use `std::uint8_t` as the
accumulator:

```cpp
std::uint8_t count = 0;
for (std::uint8_t value : values) {
  if ((value % 2) == 0) {
    ++count;
  }
}
```

That narrower accumulator can let the compiler use packed byte additions
instead of widening every predicate result into much larger lanes. On x86-64,
the interesting contrast is a loop that uses byte-lane operations such as
`vpaddb` versus one that widens to 64-bit lanes. On AArch64, inspect whether the
compiler keeps more work in byte/vector lanes before doing the final horizontal
reduction.

### Planned Implementations

- `StdCountIfEven`: uses `std::count_if` as the ordinary STL baseline.
- `CountIfEvenU8`: uses an explicit `std::uint8_t` accumulator when the result
  is known to fit in one byte.
- `CountIfEvenU16` or `CountIfEvenU32`: extends the same idea to wider known
  result ranges.

### Planned Tests and Benchmarks

The correctness tests should generate buffers whose even-value count is known
to fit in the chosen accumulator type. The benchmark should compare the STL
baseline against the narrow-accumulator variants across several buffer sizes.

This experiment has a stronger precondition than the fake-length search:
overflow is part of the design space. If the true count can exceed the
accumulator range, the narrow version either needs a larger accumulator or must
be interpreted as a modulo count.

## How to Run

### Correctness

```bash
bazel test //topics/005-compiler-auto-vectorization-experiments:scan_test
```

### Benchmarks

```bash
bazel run -c opt //topics/005-compiler-auto-vectorization-experiments:scan_bench
```

To focus on the string scan:

```bash
bazel run -c opt //topics/005-compiler-auto-vectorization-experiments:scan_bench -- \
  --benchmark_filter="Strlen"
```

## What to Observe

This experiment is compiler-sensitive.

- GCC 15.1 and newer may auto-vectorize the bounded helper and, after inlining,
  the fake-length `strlen` variant.
- Older GCC and Clang may leave both scans scalar, or may optimize only the
  bounded helper.
- Apple Clang on Apple Silicon may show little or no win for the fake-length
  variant. That is still useful data: the point of this topic is to compare
  compiler behavior, not to bake in one expected result.
- For the narrow-accumulator count experiment, Clang may produce a noticeably
  wider packed loop when the accumulator type is explicit. GCC behavior is
  version-sensitive; GCC 15 is expected to be more capable than GCC 9-14 for
  some `count_if` even/odd patterns.

When investigating generated code, build with a specific compiler and inspect
the object or assembly for vector compares and vector loads. On x86-64, look
for instructions such as `vpcmpeqb`; on AArch64, look for vector byte compare
instructions such as `cmeq`.

## Caveats

- The fake-length functions require that the terminator condition is eventually
  true. Calling them on unterminated data has the same practical failure mode as
  calling `strlen` on an unterminated string.
- `SIZE_MAX` works only with a strict `< len` loop condition. A `<= len` loop
  would make the bound always true for `SIZE_MAX` after unsigned wraparound
  reasoning, defeating the point of the trick.
- Benchmark numbers depend heavily on compiler version, target CPU, alignment,
  libc recognition, and benchmark data size.

## References

- Original article: "Auto-vectorizing operations on buffers of unknown length"
- Godbolt from the article: https://godbolt.org/z/Yc1vMbqqP
- Original article: "Improving on `std::count_if()`'s auto-vectorization"
