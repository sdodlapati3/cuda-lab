# Day 5: Capstone Implementation

## Objective
Implement your capstone project based on Day 4's plan.

## Implementation Checklist

### Phase 1: Setup (30 mins)
- [ ] Create project directory structure
- [ ] Set up CMakeLists.txt
- [ ] Create main.cu skeleton
- [ ] Verify build works

### Phase 2: Core Kernels (2-3 hours)
- [ ] Implement first kernel (naive version)
- [ ] Test for correctness
- [ ] Implement second kernel if applicable
- [ ] Test kernel integration

### Phase 3: Optimization (1-2 hours)
- [ ] Profile with nsys/ncu
- [ ] Identify bottlenecks
- [ ] Apply optimizations
- [ ] Re-profile to verify

### Phase 4: Benchmarking (30 mins)
- [ ] Implement CPU baseline
- [ ] Create timing infrastructure
- [ ] Run comparative benchmarks
- [ ] Record results

## Code Quality Guidelines

### 1. Error Handling
```cpp
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
```

### 2. Timing
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
// ... kernel calls ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

### 3. Verification
```cpp
// Compare GPU result with CPU result
bool verify(float* gpu, float* cpu, int n, float tolerance) {
    for (int i = 0; i < n; i++) {
        if (fabs(gpu[i] - cpu[i]) > tolerance) {
            printf("Mismatch at %d: GPU=%.6f, CPU=%.6f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}
```

## Common Issues and Solutions

### Issue 1: Kernel doesn't produce correct results
- Add printf debugging in kernel (use single thread)
- Verify memory sizes and offsets
- Check for race conditions

### Issue 2: Performance is poor
- Profile to find bottleneck
- Check memory access patterns
- Verify occupancy

### Issue 3: Out of memory
- Reduce problem size for testing
- Use streaming for large data
- Check for memory leaks

## Tips for Success
1. Start simple, add complexity
2. Test frequently, commit often
3. Profile before optimizing
4. Document as you go
