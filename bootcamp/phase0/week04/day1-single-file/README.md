# Day 1: Single-File Project Template

## Overview

The simplest CUDA project structure for quick experiments and learning.

## When to Use

- Quick prototyping
- Learning new CUDA features
- Small utilities
- Code examples

## Structure

```
my-project/
├── CMakeLists.txt
├── build.sh
├── main.cu
└── README.md
```

## Quick Start

```bash
./build.sh
./build/my_cuda_app
```

## Features

- Single .cu file with main()
- Simple CMake configuration
- Error checking macros
- Timer for benchmarking
- Multi-arch support
