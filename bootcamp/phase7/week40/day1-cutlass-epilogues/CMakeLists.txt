cmake_minimum_required(VERSION 3.18)
project(week40_day1 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(cutlass_intro cutlass_intro.cu)
