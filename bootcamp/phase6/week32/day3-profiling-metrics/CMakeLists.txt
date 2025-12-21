cmake_minimum_required(VERSION 3.18)
project(week32_day3 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(profiling_metrics profiling_metrics.cu)
