cmake_minimum_required(VERSION 3.18)
project(spatial_hash CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(spatial_hash spatial_hash.cu)
target_compile_options(spatial_hash PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
