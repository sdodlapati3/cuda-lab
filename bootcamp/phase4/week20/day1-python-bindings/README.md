# Day 1: Python Bindings for CUDA

## Learning Objectives
- Understand different methods for Python-CUDA interop
- Create bindings using pybind11
- Use ctypes for simple function wrapping
- Handle NumPy arrays efficiently

## Methods for Python Bindings

### 1. CuPy (Easiest)
- Drop-in NumPy replacement for GPU
- Write custom kernels in strings
- No compilation step

### 2. Numba CUDA (Easy)
- JIT compilation of Python to CUDA
- Good for prototyping
- Some limitations vs raw CUDA

### 3. pybind11 (Flexible)
- Full C++/CUDA integration
- Best performance
- More setup required

### 4. ctypes (Simple)
- Load shared libraries directly
- No additional dependencies
- Manual type handling

## pybind11 Example Structure
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Your CUDA kernel wrapper
void vector_add_cuda(float* a, float* b, float* c, int n);

py::array_t<float> vector_add(py::array_t<float> a, py::array_t<float> b) {
    // Get buffer info
    auto buf_a = a.request();
    auto buf_b = b.request();
    
    // Create output array
    auto result = py::array_t<float>(buf_a.size);
    auto buf_result = result.request();
    
    // Call CUDA function
    vector_add_cuda(
        static_cast<float*>(buf_a.ptr),
        static_cast<float*>(buf_b.ptr),
        static_cast<float*>(buf_result.ptr),
        buf_a.size
    );
    
    return result;
}

PYBIND11_MODULE(cuda_ops, m) {
    m.def("vector_add", &vector_add, "Add two vectors on GPU");
}
```

## Key Considerations
- Memory management (who owns the data?)
- Data transfer overhead
- GIL release for long operations
- Error handling across language boundaries

## Exercises
1. Create a simple vector addition binding
2. Handle multi-dimensional NumPy arrays
3. Benchmark Python vs direct CUDA call
4. Add proper error handling
