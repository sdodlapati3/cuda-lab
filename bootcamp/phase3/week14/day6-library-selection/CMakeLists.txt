cmake_minimum_required(VERSION 3.18)
project(LibraryComparison LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

find_package(CUDAToolkit REQUIRED)

add_executable(library_comparison library_comparison.cu)
target_link_libraries(library_comparison PRIVATE CUDA::cublas)
