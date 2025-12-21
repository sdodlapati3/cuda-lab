cmake_minimum_required(VERSION 3.18)
project(particle_system CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(particle_system particle_system.cu)
target_compile_options(particle_system PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
