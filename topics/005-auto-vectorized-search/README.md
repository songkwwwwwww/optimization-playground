# 005: Auto-vectorized Search With a Fake Length

This topic studies a compiler optimization trick from the article
"Auto-vectorizing operations on buffers of unknown length".

The core question is: can we make a loop that normally has no known trip count,
such as a `strlen`-style scan, look like a bounded buffer search so the compiler
is more willing to auto-vectorize it?

## Idea

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

## Implementations

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

## How to Run

### Correctness

```bash
bazel test //topics/005-auto-vectorized-search:scan_test
```

### Benchmarks

```bash
bazel run -c opt //topics/005-auto-vectorized-search:scan_bench
```

To focus on the string scan:

```bash
bazel run -c opt //topics/005-auto-vectorized-search:scan_bench -- \
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
