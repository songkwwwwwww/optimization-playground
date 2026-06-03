# Design Doc: Fixed-size Object Pool

Status: Draft  
Author: optimization-playground  
Last updated: 2026-05-23

## Summary

This document proposes and describes `memory_pool::ObjectPool<T,
BlockCapacity>`, a fixed-size object pool for studying allocator behavior,
object lifetime management, locality, and allocation overhead in C++23.

The pool is not intended to replace a general-purpose allocator. It is a small
experimental component used by `topics/004-memory-pool` to compare predictable
same-sized object reuse against ordinary `new`/`delete`.

## Background

Dynamic allocation is often too expensive or too unpredictable for hot paths.
The allocator must find storage, maintain metadata, handle fragmentation, and
possibly synchronize with other threads. For low-latency systems, parsers,
queues, order books, simulation engines, and temporary object pipelines, the
ideal allocation pattern is often narrower:

- allocate many objects of the same type
- destroy objects individually
- reuse recently freed slots
- avoid returning to the system allocator on every operation
- preserve object construction and destruction semantics

`ObjectPool` targets this narrower pattern. It allocates storage in blocks,
threads free slots through a singly linked free list, and constructs objects in
place only when the user calls `New`.

## Goals

- Provide a clear educational implementation of a free-list object pool.
- Keep the allocation and deallocation fast path O(1) after at least one block
  has been allocated.
- Correctly manage C++ object lifetime with `std::construct_at` and
  `std::destroy_at`.
- Support move-only and non-trivially destructible types.
- Preserve alignment for normal and over-aligned object types.
- Add unit tests that cover lifetime, reuse, growth, move-only values, and
  alignment.
- Add benchmarks that compare pooled allocation with `new`/`delete` for batch
  and interleaved allocation patterns.

## Non-goals

- This is not a `std::allocator` or `std::pmr::memory_resource`
  implementation.
- This does not provide thread safety. Callers must externally synchronize if
  they share a pool across threads.
- This does not validate that a pointer passed to `Delete` belongs to the pool.
- This does not detect double free, use-after-free, or mismatched-pool
  deallocation in release builds.
- This does not shrink individual blocks or return partially unused blocks to
  the upstream allocator.
- This does not optimize for variable-sized allocations.

## API

The public API is intentionally small:

```cpp
namespace memory_pool {

template <typename T, std::size_t BlockCapacity = 1024>
class ObjectPool {
 public:
  ObjectPool();
  ObjectPool(const ObjectPool&) = delete;
  ObjectPool& operator=(const ObjectPool&) = delete;
  ObjectPool(ObjectPool&&) = delete;
  ObjectPool& operator=(ObjectPool&&) = delete;
  ~ObjectPool();

  template <typename... Args>
  T* New(Args&&... args);

  void Delete(T* object) noexcept(std::is_nothrow_destructible_v<T>);
  void Clear() noexcept(std::is_nothrow_destructible_v<T>);

  std::size_t live_count() const noexcept;
  std::size_t free_count() const noexcept;
  std::size_t block_count() const noexcept;
  std::size_t capacity() const noexcept;
};

}  // namespace memory_pool
```

### API Notes

- `New` returns a raw pointer because the pool owns the backing storage, not the
  object lifetime policy.
- `Delete(nullptr)` is a no-op, matching the ergonomics of `delete`.
- `Clear` destroys all live objects and releases all blocks.
- Copy and move operations are disabled because outstanding object pointers are
  stable only while their original pool remains in place.

## Detailed Design

### Data Model

The pool owns a vector of heap-allocated blocks:

```cpp
std::vector<std::unique_ptr<Block>> blocks_;
Slot* free_head_;
std::size_t live_count_;
std::size_t free_count_;
```

Each block contains `BlockCapacity` slots. A slot is either:

- free: its storage is reused as a `Slot* next` pointer in the free list
- occupied: its storage contains a live `T` object

Each slot also stores an `occupied` bit. This bit is not needed for the fast
path, but it lets `Clear` and the pool destructor find live objects that still
need destruction.

### Slot Layout and Alignment

Each slot is aligned to the larger of `alignof(T)` and `alignof(void*)`.

This matters because a free slot stores a `Slot*`, while an occupied slot stores
a `T`. For small objects such as `int`, aligning only to `alignof(T)` would be
insufficient for the free-list pointer. For over-aligned objects, aligning only
to a pointer would be insufficient for `T`.

```cpp
static constexpr std::size_t kSlotAlignment =
    alignof(T) > alignof(void*) ? alignof(T) : alignof(void*);

struct alignas(kSlotAlignment) Slot {
  union {
    Slot* next;
    std::byte storage[sizeof(T)];
  };
  bool occupied;
};
```

### Allocation Path

`New(args...)` follows this sequence:

1. If `free_head_` is null, allocate a new block.
2. Pop the first slot from the free list.
3. Construct `T` in the slot storage with `std::construct_at`.
4. Mark the slot occupied.
5. Update counters and return the constructed object pointer.

The common path after warmup performs one free-list pop, one placement
construction, and counter updates.

### Deallocation Path

`Delete(object)` follows this sequence:

1. Return immediately if `object` is null.
2. Convert the object address back to its containing slot address.
3. Destroy the object with `std::destroy_at`.
4. Mark the slot free.
5. Push the slot onto `free_head_`.
6. Update counters.

The implementation relies on the object being located at the start of `Slot`.
This is true because the active object is constructed in the union storage,
which starts at offset zero in the slot.

### Growth Policy

The pool grows one block at a time. When a block is added, all of its slots are
pushed onto the free list.

The current implementation uses a fixed `BlockCapacity` template parameter. This
keeps the implementation simple and moves block sizing into compile-time
configuration. It also makes benchmarks easier to interpret because block size
does not change at runtime.

### Lifetime and Destruction

The pool separates storage lifetime from object lifetime:

- block allocation creates raw storage slots
- `New` creates a `T` object inside one slot
- `Delete` destroys one `T` object and returns the slot
- `Clear` destroys all remaining live objects and releases all blocks
- `~ObjectPool` calls `Clear`

This design is meant to make lifetime operations visible in tests and easy to
inspect in a debugger.

## Correctness Invariants

- `live_count() + free_count() == capacity()` after every successful operation.
- Every slot in every block is either free or occupied, never both.
- `free_head_` points only to free slots owned by the pool.
- A live object is destroyed exactly once by `Delete`, `Clear`, or the pool
  destructor.
- Object addresses remain stable until the object is deleted or the pool is
  cleared.
- All returned object pointers satisfy `alignof(T)`.

## Error Handling

The implementation intentionally keeps error handling minimal:

- allocation failure from `std::make_unique<Block>` propagates as
  `std::bad_alloc`
- constructor exceptions from `T` propagate to the caller
- destructor exceptions are not supported in practice; `Delete` and `Clear`
  are conditionally `noexcept` only when `T` is nothrow destructible

Future debug builds could add ownership checks or double-free detection, but
those checks are not part of the first implementation.

## Performance Considerations

The expected performance win comes from avoiding a system allocator round trip
on every object allocation. Once the pool has free slots, allocation is a
pointer pop and deallocation is a pointer push.

The expected costs are:

- one `occupied` bit per slot
- possible padding from `kSlotAlignment`
- block-level allocation when the free list is empty
- no automatic release of partially empty blocks
- raw-pointer API discipline at call sites

The benchmark covers two workloads:

- **Batch allocation:** allocate many objects, then delete many objects.
- **Interleaved allocation:** allocate one object and immediately delete it.

Batch allocation tests block growth and reuse. Interleaved allocation isolates
the hot-path free-list cost once a block exists.

## Alternatives Considered

### `new` / `delete`

This is the baseline. It is general-purpose and simple, but it pays allocator
overhead for each object and may have weaker locality under churn-heavy
workloads.

### Arena Allocator

An arena can make allocation extremely cheap and support bulk reset. It is less
suited to this topic's first implementation because individual object deletion
is important for comparing free-list reuse.

### `std::pmr::memory_resource`

`std::pmr` would make the component easier to plug into STL containers. It also
adds interface complexity that can distract from the core free-list and object
lifetime mechanics. A PMR adapter is a good follow-up experiment.

### Intrusive Free List in `T`

An intrusive design could store the next pointer inside `T` itself. That avoids
some slot metadata but constrains user types and makes the educational object
lifetime story less explicit.

## Testing Plan

Tests live in `object_pool_test.cpp` and should cover:

- construction and destruction counts for non-trivial objects
- freed slot reuse
- block growth beyond one block
- `Clear` destroying only live objects
- move-only object support
- over-aligned object alignment

Run:

```bash
bazel test //topics/004-memory-pool:object_pool_test
```

## Benchmark Plan

Benchmarks live in `object_pool_bench.cpp` and compare:

- `BM_NewDeleteBatch`
- `BM_ObjectPoolBatch`
- `BM_NewDeleteInterleaved`
- `BM_ObjectPoolInterleaved`

Run:

```bash
bazel run -c opt //topics/004-memory-pool:object_pool_bench
```

Benchmark interpretation should focus on allocation pattern, not only absolute
time. The pool should shine when objects are reused and allocation sizes are
uniform. The system allocator may remain competitive for small interleaved
patterns because modern allocators already use fast per-thread caches.

## Rollout Plan

1. Add the initial `ObjectPool` implementation.
2. Add correctness tests for lifetime, reuse, growth, move-only support, and
   alignment.
3. Add allocation benchmarks against `new`/`delete`.
4. Document measured results in the topic README after running benchmarks on the
   target machine.
5. Use the implementation as a baseline for later allocator experiments.

## Future Work

- Add an RAII handle that returns objects to the pool automatically.
- Add a debug mode with ownership and double-free checks.
- Add a `std::pmr::memory_resource` adapter.
- Add a monotonic arena implementation for comparison.
- Add a thread-local pool variant for multi-threaded allocation workloads.
- Add allocation-count instrumentation.
- Measure p50, p95, p99, and max allocation latency, not only average
  throughput.

## Open Questions

- Should the pool expose `ReserveBlocks(n)` or `ReserveObjects(n)` for explicit
  warmup?
- Should `Clear` preserve allocated blocks for reuse, with a separate
  `Release` API for returning memory?
- Should the first production-like extension prioritize RAII ownership or PMR
  integration?
- Should debug safety checks be compiled in with a Bazel config flag?
