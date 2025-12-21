cmake_minimum_required(VERSION 3.18)
project(sph_fluid CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(sph_fluid sph_fluid.cu)
target_compile_options(sph_fluid PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
