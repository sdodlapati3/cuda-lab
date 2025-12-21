cmake_minimum_required(VERSION 3.18)
project(collision_detection CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(collision_detection collision_detection.cu)
target_compile_options(collision_detection PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
