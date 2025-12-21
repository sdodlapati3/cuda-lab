cmake_minimum_required(VERSION 3.18)
project(double_buffering CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(double_buffering double_buffering.cu)
target_link_libraries(double_buffering CUDA::cublas)
target_compile_options(double_buffering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ptxas-options=-v>)
