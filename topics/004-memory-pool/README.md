# Memory Pool Optimization Study

This topic studies a fixed-size object pool: a small allocator-like data
structure that trades general-purpose allocation flexibility for predictable,
cache-friendly reuse of same-sized objects.

## What This Implements

`ObjectPool<T, BlockCapacity>` stores uninitialized slots in blocks. When `New`
is called, the pool pops one slot from a free list and constructs `T` in place
with `std::construct_at`. When `Delete` is called, the pool runs the destructor
with `std::destroy_at` and returns the slot to the free list.

This is intentionally not a drop-in replacement for `std::allocator`. It is a
focused learning implementation for studying:

- object lifetime control with placement construction and explicit destruction
- free-list based allocation
- block growth to amortize upstream allocation cost
- slot reuse locality
- over-aligned object support

For the fuller design rationale, see
[docs/DESIGN_DOC.md](docs/DESIGN_DOC.md).

## Design Notes

- **Fast path:** allocation and deallocation are O(1) free-list operations once a
  block exists.
- **Growth policy:** the pool grows by `BlockCapacity` slots whenever the free
  list is empty.
- **Lifetime tracking:** each slot keeps an `occupied` bit so `Clear` and the
  destructor can destroy any still-live objects.
- **Thread safety:** this pool is single-threaded / externally synchronized.
  A later variant could add a lock, a thread-local cache, or a lock-free free
  list.

## Tradeoffs

| Approach | Strength | Cost |
| :--- | :--- | :--- |
| `new` / `delete` | General purpose, simple ownership model | Per-object allocator overhead and less predictable locality |
| `ObjectPool` | Fast reuse for many same-sized objects | Objects must be returned to the pool that created them |
| Arena allocator | Very fast bulk allocation/reset | Usually cannot destroy individual objects cheaply |

## How to Run

Run the correctness tests:

```bash
bazel test //topics/004-memory-pool:object_pool_test
```

Run the allocation benchmarks:

```bash
bazel run -c opt //topics/004-memory-pool:object_pool_bench
```

The benchmark compares batched and interleaved allocation patterns for
`new`/`delete` versus `ObjectPool`.
