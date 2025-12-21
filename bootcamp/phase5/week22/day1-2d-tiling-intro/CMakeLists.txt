cmake_minimum_required(VERSION 3.18)
project(tiling_intro CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(tiling_intro tiling_intro.cu)
target_compile_options(tiling_intro PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
