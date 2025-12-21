# Reusable CUDA Project Template
#
# Usage:
#   mkdir build && cd build
#   cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES=80 ..
#   ninja
#
# Multi-arch build:
#   cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES="80;90" ..

cmake_minimum_required(VERSION 3.18)
project(CudaProjectTemplate LANGUAGES CXX CUDA)

# ============================================================================
# Standards
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# Architecture
# ============================================================================
# Default to A100 if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
    message(STATUS "CUDA architecture defaulting to sm_80 (A100)")
endif()

# ============================================================================
# Build Configuration
# ============================================================================
# Options
option(VERBOSE_PTX "Show PTX compilation details (register/smem usage)" OFF)
option(ENABLE_PROFILING "Build with -lineinfo for profiling" ON)

# Base flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Profiling support
if(ENABLE_PROFILING)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    message(STATUS "Profiling enabled (-lineinfo)")
endif()

# Verbose PTX
if(VERBOSE_PTX)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
    message(STATUS "Verbose PTX enabled")
endif()

# ============================================================================
# Includes
# ============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# Library of utilities
# ============================================================================
add_library(cuda_utils STATIC
    src/cuda_utils.cu
)

# ============================================================================
# Main executable
# ============================================================================
add_executable(main src/main.cu)
target_link_libraries(main cuda_utils)

# ============================================================================
# Benchmarks
# ============================================================================
add_executable(benchmark 
    benchmarks/benchmark_main.cu
)
target_link_libraries(benchmark cuda_utils)

# ============================================================================
# Tests
# ============================================================================
enable_testing()
add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels cuda_utils)
add_test(NAME KernelTests COMMAND test_kernels)

# ============================================================================
# Print Configuration
# ============================================================================
message(STATUS "")
message(STATUS "=== CUDA Project Configuration ===")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "==================================")
