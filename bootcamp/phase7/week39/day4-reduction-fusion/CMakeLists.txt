cmake_minimum_required(VERSION 3.18)
project(week39_day4 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(reduction_fusion reduction_fusion.cu)
