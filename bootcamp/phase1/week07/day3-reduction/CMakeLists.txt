cmake_minimum_required(VERSION 3.18)
project(Reduction LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

add_executable(reduction reduction.cu)
add_executable(dot_product dot_product.cu)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(dot_product CUDA::cublas)
