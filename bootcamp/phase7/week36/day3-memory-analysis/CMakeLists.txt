cmake_minimum_required(VERSION 3.18)
project(week36_day3 CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(memory_analysis memory_analysis.cu)
