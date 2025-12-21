cmake_minimum_required(VERSION 3.18)
project(profiling_demo CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(profiling_demo profiling_demo.cu)
target_compile_options(profiling_demo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
