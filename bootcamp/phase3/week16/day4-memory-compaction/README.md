# Day 4: Memory Compaction

## Learning Objectives

- Understand memory fragmentation
- Implement compaction strategies
- Use stream-ordered allocations effectively

## Key Concepts

### The Fragmentation Problem

```
GPU Memory (fragmented):
[Alloc1][Free][Alloc2][Free][Free][Alloc3][Free]
         ^^^         ^^^^^^^^^^          ^^^
         Unusable small gaps

Need 4 contiguous blocks → FAILS despite having space!
```

### Strategies

1. **Pool allocators**: Allocate in fixed-size blocks
2. **Slab allocators**: Different pools for different sizes
3. **Memory compaction**: Periodically consolidate
4. **Defragmentation**: Move allocations together

### Stream-Ordered Allocation Helps

```cpp
// Old pattern: fragments easily
for (work in works) {
    cudaMalloc(&ptr, size);
    process(ptr);
    cudaFree(ptr);  // Freed at different times → fragments
}

// New pattern: memory pool reuses
for (work in works) {
    cudaMallocAsync(&ptr, size, stream);
    process(ptr, stream);
    cudaFreeAsync(ptr, stream);  // Returns to pool for reuse
}
```

### Custom Pool Allocator Pattern

```cpp
class FixedSizePool {
    void* base;
    size_t block_size;
    std::queue<void*> free_blocks;
public:
    void* alloc() { return free_blocks.pop(); }
    void free(void* p) { free_blocks.push(p); }
};
```

## Build & Run

```bash
./build.sh
./build/memory_compaction
```
