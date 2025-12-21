# Day 5: Thrust Library

## Learning Objectives

- Use Thrust's STL-like interface for GPU
- Work with device_vector
- Apply transformations and algorithms

## Key Concepts

### Thrust Basics

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

// Automatic memory management
thrust::device_vector<float> d_vec(1000000);

// STL-like algorithms
thrust::fill(d_vec.begin(), d_vec.end(), 1.0f);
float sum = thrust::reduce(d_vec.begin(), d_vec.end());
thrust::sort(d_vec.begin(), d_vec.end());
```

### Transform with Functors

```cpp
struct square {
    __host__ __device__
    float operator()(float x) { return x * x; }
};

thrust::transform(d_in.begin(), d_in.end(), d_out.begin(), square());
```

### Interop with Raw CUDA

```cpp
// Get raw pointer for CUDA kernels
float* raw_ptr = thrust::raw_pointer_cast(d_vec.data());
my_kernel<<<blocks, threads>>>(raw_ptr, d_vec.size());

// Wrap existing device memory
thrust::device_ptr<float> d_ptr(raw_cuda_ptr);
```

### Decision: Thrust vs CUB

| Use Thrust | Use CUB |
|------------|---------|
| Rapid prototyping | Inside kernels |
| Simple transforms | Maximum control |
| Automatic memory | Custom algorithms |

## Build & Run

```bash
./build.sh
./build/thrust_demo
```
