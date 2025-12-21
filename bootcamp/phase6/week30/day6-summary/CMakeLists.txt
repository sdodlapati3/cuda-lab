cmake_minimum_required(VERSION 3.18)
project(week30_day6 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(custom_ops_summary custom_ops_summary.cu)
