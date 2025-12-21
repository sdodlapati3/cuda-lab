cmake_minimum_required(VERSION 3.18)
project(tiling_summary CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(tiling_summary tiling_summary.cu)
target_link_libraries(tiling_summary CUDA::cublas)
target_compile_options(tiling_summary PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)
