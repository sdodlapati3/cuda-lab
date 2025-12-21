cmake_minimum_required(VERSION 3.18)
project(memory_analysis CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(memory_analysis memory_analysis.cu)
target_compile_options(memory_analysis PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
target_link_libraries(memory_analysis CUDA::cublas)
