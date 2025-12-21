# CUDA Project Template

A clean, reusable template for CUDA projects.

## Quick Start

```bash
chmod +x build.sh
./build.sh 80     # Build for A100
./build/main      # Run
./build/benchmark # Benchmark
ctest --test-dir build  # Run tests
```

## Structure

```
.
├── CMakeLists.txt      # Build configuration
├── build.sh            # Quick build script
├── include/            # Headers
│   └── cuda_utils.cuh  # Error checking, timing, device info
├── src/                # Source files
│   ├── main.cu         # Entry point
│   └── cuda_utils.cu   # Utilities implementation
├── benchmarks/         # Performance benchmarks
└── tests/              # Correctness tests
```

## Features

- **Modern CMake** (3.18+) with CUDA as first-class language
- **Multi-architecture** support via CMAKE_CUDA_ARCHITECTURES
- **RAII wrappers** for device memory (DeviceBuffer)
- **CUDA event timing** for accurate benchmarks
- **Error checking** macros (CUDA_CHECK, CUDA_KERNEL_CHECK)
- **Device info** query (get_gpu_info())

## Usage

### Adding a new kernel

1. Create `kernels/my_kernel.cu`
2. Add to CMakeLists.txt
3. Add tests in `tests/`
4. Add benchmarks in `benchmarks/`

### Profiling

```bash
# Build with profiling support (default)
./build.sh 80 RelWithDebInfo

# Profile with Nsight Compute
ncu --set full ./build/main
```

## Customization

Edit CMakeLists.txt to:
- Add new source files
- Change default architecture
- Add libraries (cuBLAS, etc.)
