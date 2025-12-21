# Day 5: Vote Functions

## Learning Objectives

- Use warp vote functions for collective decisions
- Understand ballot, any, and all operations
- Apply votes for predicate handling

## Key Concepts

### Vote Functions

```cpp
// Returns 1 if ANY thread in warp has predicate=true
int __any_sync(mask, predicate);

// Returns 1 if ALL threads in warp have predicate=true
int __all_sync(mask, predicate);

// Returns 32-bit mask where bit i is set if lane i has predicate=true
unsigned int __ballot_sync(mask, predicate);
```

### Visual Examples

```
Thread predicates: [T F T T F F T F | T T T T T T T T | ...]

__any_sync():   Returns 1 (at least one true)
__all_sync():   Returns 0 (not all true)
__ballot_sync(): Returns 0b...11111111_01000111 (bitmask)
```

### Common Patterns

```cpp
// Count matching elements
unsigned int mask = __ballot_sync(FULL_MASK, predicate);
int count = __popc(mask);  // Population count

// Get active lane count
int active_count = __popc(__activemask());

// Check if entire warp matches
if (__all_sync(FULL_MASK, condition)) {
    // All 32 threads match - take fast path
}
```

## Build & Run

```bash
./build.sh
./build/vote_demo
```
