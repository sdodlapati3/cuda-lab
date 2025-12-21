cmake_minimum_required(VERSION 3.18)
project(BottleneckAnalysis LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

add_executable(bottleneck_demo src/bottleneck_demo.cu)
add_executable(optimization_examples src/optimization_examples.cu)
