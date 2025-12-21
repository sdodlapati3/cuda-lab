cmake_minimum_required(VERSION 3.18)
project(tiled_nbody CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(tiled_nbody tiled_nbody.cu)
target_compile_options(tiled_nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
