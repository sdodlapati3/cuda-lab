cmake_minimum_required(VERSION 3.18)
project(week32_day1 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(triton_basics triton_basics.cu)
