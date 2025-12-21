cmake_minimum_required(VERSION 3.18)
project(Roofline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

# Include the benchmark harness from Day 4
include_directories(${CMAKE_SOURCE_DIR}/../day4-benchmark-harness/include)

# Main measurement executable
add_executable(roofline_measure
    src/bandwidth_test.cu
    src/compute_test.cu
    src/example_kernels.cu
    src/main.cu
)
