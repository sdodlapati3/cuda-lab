cmake_minimum_required(VERSION 3.18)
project(multi_gpu CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(multi_gpu multi_gpu.cu)
target_compile_options(multi_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
