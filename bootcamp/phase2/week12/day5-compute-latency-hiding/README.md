# Day 5: Compute Latency Hiding

## Learning Objectives

- Reduce instruction dependencies
- Use loop unrolling for ILP
- Understand software pipelining

## Key Concepts

### Instruction Dependencies

```
Dependency chain (serialized):
  a = x * 2      // 4 cycles
  b = a + 1      // 4 cycles (waits for a)
  c = b * 3      // 4 cycles (waits for b)
  Total: 12 cycles

Independent ops (parallel):
  a = x * 2      // 4 cycles ─┐
  b = y + 1      // 4 cycles ─┼─ All in parallel
  c = z * 3      // 4 cycles ─┘
  Total: 4 cycles
```

### Loop Unrolling

```cpp
// Before: Long dependency chain
for (int i = 0; i < 4; i++) {
    sum += data[i];  // Each waits for previous
}

// After: Independent partial sums
float s0 = data[0], s1 = data[1];
float s2 = data[2], s3 = data[3];
sum = (s0 + s1) + (s2 + s3);  // More parallelism
```

### Register Reuse

```cpp
// Bad: Reads from same register
for (int i = 0; i < n; i++) {
    result = result * data[i];  // Dependency on result
}

// Better: Multiple accumulators
float r0 = 1.0f, r1 = 1.0f;
for (int i = 0; i < n; i += 2) {
    r0 *= data[i];      // Independent
    r1 *= data[i + 1];  // Independent
}
result = r0 * r1;
```

## Build & Run

```bash
./build.sh
./build/compute_latency
```
