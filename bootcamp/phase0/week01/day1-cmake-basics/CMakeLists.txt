# Minimum CMake version that supports modern CUDA
cmake_minimum_required(VERSION 3.18)

# Project name and languages
# CUDA is a first-class language in CMake 3.18+
project(HelloGPU LANGUAGES CXX CUDA)

# Set C++ standard for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect GPU architecture if not specified
# Common values: 70 (V100), 80 (A100), 86 (RTX 3090), 89 (RTX 4090), 90 (H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
    message(STATUS "CUDA architecture not specified, defaulting to sm_80")
endif()

# Enable optimizations and debug info for profiling
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")  # Source mapping for profilers

# Optional: Verbose PTX assembly generation
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# Create executable
add_executable(hello_gpu src/hello_gpu.cu)

# Set CUDA-specific properties
set_target_properties(hello_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # Needed for device linking
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "===========================")
