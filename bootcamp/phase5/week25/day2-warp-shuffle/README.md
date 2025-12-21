# Week 25, Day 2: Warp Shuffle Basics

## Objective
Learn warp shuffle operations for register-to-register communication.

## Key Concepts
- __shfl_sync(): Get value from any lane
- __shfl_xor_sync(): Butterfly pattern
- __shfl_up_sync() / __shfl_down_sync(): Shift patterns
- No shared memory needed for intra-warp communication

## Shuffle Operations
```cpp
// Broadcast from lane srcLane
val = __shfl_sync(0xFFFFFFFF, val, srcLane);

// XOR butterfly (swap with lane ^ xorMask)
val = __shfl_xor_sync(0xFFFFFFFF, val, xorMask);

// Shift operations
val = __shfl_up_sync(0xFFFFFFFF, val, delta);
val = __shfl_down_sync(0xFFFFFFFF, val, delta);
```

## Use Cases
- Broadcast A values across warp
- Parallel reduction
- Transpose small matrices in registers
