cmake_minimum_required(VERSION 3.18)
project(col_tiled_gemm CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(col_tiled col_tiled.cu)
target_link_libraries(col_tiled CUDA::cublas)
target_compile_options(col_tiled PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)
