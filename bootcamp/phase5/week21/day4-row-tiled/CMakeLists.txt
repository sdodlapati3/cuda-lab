cmake_minimum_required(VERSION 3.18)
project(row_tiled_gemm CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(row_tiled row_tiled.cu)
target_link_libraries(row_tiled CUDA::cublas)
target_compile_options(row_tiled PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)
