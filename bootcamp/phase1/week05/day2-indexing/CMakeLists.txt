cmake_minimum_required(VERSION 3.18)
project(ThreadIndexing LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

add_executable(index_1d index_1d.cu)
add_executable(index_2d index_2d.cu)
add_executable(index_3d index_3d.cu)
