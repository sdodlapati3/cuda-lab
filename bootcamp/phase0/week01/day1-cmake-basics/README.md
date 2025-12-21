# Day 1: CMake + Ninja Basics

## Why CMake for CUDA?

Before CMake 3.18, building CUDA required `find_package(CUDA)` and the old `cuda_add_executable()` commands. Modern CMake treats CUDA as a first-class language—just like C++.

**Old way (don't use):**
```cmake
find_package(CUDA REQUIRED)
cuda_add_executable(myapp main.cu)
```

**New way (use this):**
```cmake
project(MyApp LANGUAGES CXX CUDA)
add_executable(myapp main.cu)
```

## Quick Start

```bash
chmod +x build.sh
./build.sh        # Builds for sm_80 (A100)
./build/hello_gpu # Run it
```

## Understanding the CMakeLists.txt

### 1. Minimum Version
```cmake
cmake_minimum_required(VERSION 3.18)
```
CUDA as a first-class language requires CMake 3.18+.

### 2. Enable CUDA Language
```cmake
project(HelloGPU LANGUAGES CXX CUDA)
```
This tells CMake to look for `nvcc` and set up CUDA compilation.

### 3. Architecture Selection
```cmake
set(CMAKE_CUDA_ARCHITECTURES 80)
```
| Value | GPU |
|-------|-----|
| 70 | V100 |
| 75 | Turing (T4, RTX 2080) |
| 80 | A100, A10 |
| 86 | RTX 3090, A40 |
| 89 | RTX 4090, L40 |
| 90 | H100 |

### 4. Optimization Flags
```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
```
- `-O3`: Maximum optimization
- `-lineinfo`: Preserve source info for profilers (Nsight)

## Exercises

1. **Multi-architecture build:** Modify CMakeLists.txt to build for both sm_80 and sm_90
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES 80 90)
   ```

2. **Add verbose PTX:** See register usage and shared memory
   ```cmake
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
   ```

3. **Create a library:** Split into `libgpu.a` and link against it

## Common Issues

**"No CMAKE_CUDA_COMPILER could be found"**
→ nvcc not in PATH. Run `module load cuda` or set `export PATH=/usr/local/cuda/bin:$PATH`

**"sm_XX is not supported"**
→ Your CUDA toolkit is too old for that architecture. Update toolkit or use older sm.

## Next Steps

Tomorrow we'll dive deep into compiler flags and understand what each optimization level does to your code.
