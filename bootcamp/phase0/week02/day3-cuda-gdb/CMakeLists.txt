cmake_minimum_required(VERSION 3.18)
project(CudaGdb LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# CRITICAL: -G enables device debugging
# Without it, you can't set breakpoints in kernels
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -lineinfo")

add_executable(debug_example src/debug_example.cu)
add_executable(debug_reduction src/debug_reduction.cu)
