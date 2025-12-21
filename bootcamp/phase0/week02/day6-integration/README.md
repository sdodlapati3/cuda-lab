# Day 6: Debug Workflow Integration

## What You'll Learn

- Systematic debugging workflow
- Choose the right tool for each problem
- Build debug-friendly code from the start
- Create reusable debug infrastructure

## The Debug Workflow

```
1. Reproduce → 2. Classify → 3. Tool → 4. Fix → 5. Verify
```

### Step 1: Reproduce
- Minimal test case
- Deterministic reproduction
- Document the failure

### Step 2: Classify the Problem

| Symptom | Likely Cause | First Tool |
|---------|--------------|------------|
| Wrong results (sometimes) | Race condition | compute-sanitizer --racecheck |
| Wrong results (always) | Logic error | cuda-gdb |
| Crash/hang | Memory error | compute-sanitizer --memcheck |
| cuda error | API misuse | Error checking |
| Slow execution | Performance | Nsight Systems |

### Step 3: Choose the Right Tool

```
compute-sanitizer:
├── --racecheck    # Data races
├── --memcheck     # Memory errors (default)
├── --initcheck    # Uninitialized access
└── --synccheck    # Sync errors

cuda-gdb:
├── Breakpoints    # Stop at location
├── Step           # Line by line
├── Print          # Inspect values
└── Thread focus   # Specific thread

Nsight Systems:
├── Timeline       # Execution flow
├── Bottlenecks    # Performance issues
└── GPU metrics    # Utilization
```

### Step 4: Fix
- Single change at a time
- Understand WHY it's wrong
- Document the fix

### Step 5: Verify
- Run with tools again
- Add test case
- Add assertions/checks

## Debug-Friendly Code

### Always Include Error Checking
```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
```

### Add Debug Mode
```cpp
#ifdef DEBUG
    #define CUDA_CHECK_KERNEL() do { \
        CUDA_CHECK(cudaGetLastError()); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)
#else
    #define CUDA_CHECK_KERNEL() ((void)0)
#endif
```

### Use NVTX Markers
```cpp
void my_operation() {
    nvtxRangePush("my_operation");
    // ...
    nvtxRangePop();
}
```

### Add Assertions
```cpp
__device__ void kernel_function(int idx, int n) {
    assert(idx >= 0 && idx < n);  // Device assertion
    // ...
}
```

## Debug Build Configuration

### CMakeLists.txt
```cmake
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -lineinfo")
    add_compile_definitions(DEBUG=1)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
endif()
```

### Usage
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..   # Debug build
cmake -DCMAKE_BUILD_TYPE=Release .. # Release build
```

## Quick Reference

```bash
# Race conditions
compute-sanitizer --tool racecheck ./my_app

# Memory errors
compute-sanitizer --tool memcheck ./my_app

# Uninitialized memory
compute-sanitizer --tool initcheck ./my_app

# Interactive debugging
cuda-gdb ./my_app

# Timeline profiling
nsys profile -o report ./my_app
nsys stats report.nsys-rep

# All in one debug session
compute-sanitizer ./my_app && cuda-gdb ./my_app
```

## Exercises

1. Debug the buggy application in src/
2. Practice the systematic workflow
3. Build your own debug header
4. Profile and optimize the demo
