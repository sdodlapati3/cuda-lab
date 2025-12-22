cmake_minimum_required(VERSION 3.18)
project(week40_day5 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(week_summary week_summary.cu)
