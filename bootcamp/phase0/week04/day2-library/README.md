# Day 2: CUDA Library Template

## Overview

A template for creating reusable CUDA libraries that can be linked into other projects.

## When to Use

- Building reusable GPU kernels
- Creating a CUDA-accelerated library
- Separating GPU code from application logic
- Sharing code across multiple projects

## Structure

```
my-cuda-lib/
├── CMakeLists.txt
├── build.sh
├── README.md
├── include/
│   └── mylib/
│       ├── mylib.h          # Public C++ API
│       └── kernels.cuh      # CUDA kernel declarations (internal)
├── src/
│   ├── mylib.cu             # Library implementation
│   └── kernels.cu           # Kernel implementations
└── examples/
    └── example.cpp          # Usage example
```

## Features

- Clean separation of interface and implementation
- Header-only public API
- Static or shared library output
- Example application demonstrating usage
