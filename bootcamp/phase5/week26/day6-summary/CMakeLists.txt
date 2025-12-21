cmake_minimum_required(VERSION 3.18)
project(week26_day6 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(tc_summary tc_summary.cu)
