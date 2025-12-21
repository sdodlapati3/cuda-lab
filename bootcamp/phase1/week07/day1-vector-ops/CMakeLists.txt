cmake_minimum_required(VERSION 3.18)
project(VectorOps LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

add_executable(vector_add vector_add.cu)
add_executable(saxpy saxpy.cu)

# Optional: Link cuBLAS for comparison
find_package(CUDAToolkit REQUIRED)
target_link_libraries(saxpy CUDA::cublas)
