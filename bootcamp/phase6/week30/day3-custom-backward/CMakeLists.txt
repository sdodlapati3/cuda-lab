cmake_minimum_required(VERSION 3.18)
project(week30_day3 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(custom_backward custom_backward.cu)
