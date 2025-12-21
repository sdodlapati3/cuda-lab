cmake_minimum_required(VERSION 3.18)
project(week32_day4 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(scaling_strategies scaling_strategies.cu)
