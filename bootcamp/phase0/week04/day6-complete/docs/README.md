# MyProject Documentation

## Overview

MyProject is a CUDA library providing optimized GPU operations.

## Quick Start

```bash
# Build
./build.sh

# Run application
./build/myapp

# Run tests
./build/tests

# Run benchmarks
./build/benchmarks
```

## API Reference

### Initialization

```cpp
#include "myproject/myproject.h"

// Initialize with default device
myproject::initialize();

// Initialize with specific device
myproject::initialize(1);

// Cleanup when done
myproject::cleanup();
```

### Vector Operations

```cpp
// Vector addition: c = a + b
myproject::vector_add(c, a, b, n);

// Vector scale: out = in * scalar
myproject::vector_scale(out, in, 2.0f, n);

// Reduction
float sum = myproject::reduce_sum(data, n);
```

### Matrix Operations

```cpp
// Matrix multiply: C = A * B
myproject::matmul(C, A, B, M, N, K);
```

## Building

Requirements:
- CMake 3.18+
- CUDA 11.0+
- C++17 compiler

```bash
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
make
```

## License

MIT License - see LICENSE file.
