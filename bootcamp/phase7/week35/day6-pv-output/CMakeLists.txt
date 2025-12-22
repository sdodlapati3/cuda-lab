cmake_minimum_required(VERSION 3.18)
project(week35_day6 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(pv_output pv_output.cu)
