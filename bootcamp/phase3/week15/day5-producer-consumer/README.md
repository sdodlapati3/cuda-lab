# Day 5: Producer-Consumer Fusion

## Learning Objectives

- Chain operations through registers
- Implement warp-level data passing
- Understand register vs shared memory trade-offs

## Key Concepts

### Producer-Consumer Pattern

When one operation produces data that another immediately consumes:

```cpp
// Producer-consumer in registers (fastest)
float a = load(input[idx]);
float b = producer_op(a);      // Result in register
float c = consumer_op(b);      // Uses register directly
store(output[idx], c);
```

### Warp-Level Producer-Consumer

```cpp
// Producer writes, consumer in same warp reads
float produced = produce(input[lane]);

// Shuffle to consumer lanes
float consumed = __shfl_sync(FULL_MASK, produced, source_lane);

// Consumer uses the data
float result = consume(consumed);
```

### Shared Memory Handoff

```cpp
__shared__ float buffer[BLOCK_SIZE];

// Producer phase
buffer[tid] = produce(input[idx]);
__syncthreads();

// Consumer phase
float result = consume(buffer[...]);
```

### When to Use Each

| Pattern | Latency | Use When |
|---------|---------|----------|
| Register | Lowest | Same thread produces/consumes |
| Warp shuffle | Low | Different lanes, same warp |
| Shared memory | Medium | Different threads, same block |

## Build & Run

```bash
./build.sh
./build/producer_consumer
```
