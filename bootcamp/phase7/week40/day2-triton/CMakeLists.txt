cmake_minimum_required(VERSION 3.18)
project(week40_day2 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(triton_overview triton_overview.cu)
