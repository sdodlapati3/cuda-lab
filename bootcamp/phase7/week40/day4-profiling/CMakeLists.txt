cmake_minimum_required(VERSION 3.18)
project(week40_day4 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(profiling profiling.cu)
