cmake_minimum_required(VERSION 3.18)
project(week31_day3 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(dynamic_batching dynamic_batching.cu)
