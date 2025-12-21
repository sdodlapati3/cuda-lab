cmake_minimum_required(VERSION 3.18)
project(ComputeSanitizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Debug build with line info for better error messages
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")

add_executable(race_example src/race_example.cu)
add_executable(race_fixed src/race_fixed.cu)
