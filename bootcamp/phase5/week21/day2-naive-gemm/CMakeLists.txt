cmake_minimum_required(VERSION 3.18)
project(naive_gemm CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(naive_gemm naive_gemm.cu)
target_compile_options(naive_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
target_link_libraries(naive_gemm CUDA::cublas)
