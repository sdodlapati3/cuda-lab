cmake_minimum_required(VERSION 3.18)
project(gemm_setup CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(gemm_setup gemm_setup.cu)
target_compile_options(gemm_setup PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(gemm_setup CUDA::cublas)
