cmake_minimum_required(VERSION 3.18)
project(Registers LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Show register usage during compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo --ptxas-options=-v")

add_executable(register_demo register_demo.cu)
add_executable(spill_demo spill_demo.cu)
