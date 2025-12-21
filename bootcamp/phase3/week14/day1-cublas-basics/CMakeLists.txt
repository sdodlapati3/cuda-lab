cmake_minimum_required(VERSION 3.18)
project(CublasBasics LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(cublas_basics cublas_basics.cu)
target_link_libraries(cublas_basics CUDA::cublas)
