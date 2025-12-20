# Exercise 04: Warp-Aggregated Atomics

## Objective
Implement warp-aggregated atomic operations to reduce contention on global memory.

## Background
When many threads perform atomic operations on the same memory location, contention causes serialization. Warp-aggregated atomics use Cooperative Groups to:
1. Count active threads in a warp
2. Have only ONE thread perform the atomic (adding the warp count)
3. Distribute results back to all threads

This reduces atomic operations from 32 per warp to just 1!

## Requirements
1. Use `cg::coalesced_threads()` to get active threads
2. Leader thread performs single atomic with `active.size()`
3. Broadcast result using `active.shfl()`
4. Return correct unique value for each thread

## Expected Speedup
- 5-30x depending on contention level

## Testing
```bash
make
./test.sh
```
