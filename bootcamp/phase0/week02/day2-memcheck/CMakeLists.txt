cmake_minimum_required(VERSION 3.18)
project(MemoryErrors LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Debug build for accurate line numbers
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")

add_executable(memory_errors src/memory_errors.cu)
add_executable(memory_fixed src/memory_fixed.cu)
add_executable(memory_leaks src/memory_leaks.cu)
